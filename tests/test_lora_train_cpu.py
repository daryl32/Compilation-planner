"""train_lora end to end on the CPU with a tiny random Qwen3-VL and a stand-in processor — the
loop, batching, out-of-memory fallback, checkpoints and resume. Needs torch / transformers / peft
(not in CI's requirements, so it's skipped there)."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")
pytest.importorskip("modal")

from tests.test_trial_logic import _fake_trial  # noqa: E402
import trial_logic as TL  # noqa: E402

MARK = [250, 251, 252]


class _Tok:
    padding_side = "right"

    def encode(self, text, add_special_tokens=False):
        return list(MARK)


class _Proc:
    """Prompt = 20 tokens + marker; answer = one token per character class of the target (≥3)."""
    tokenizer = _Tok()

    def apply_chat_template(self, convs, tokenize=True, return_dict=True, return_tensors="pt", padding=True):
        rows = []
        for conv in convs:
            ans = conv[1]["content"][0]["text"]
            rows.append([5] * 20 + MARK + [10 + (len(ans) + k) % 200 for k in range(3 + len(ans) % 4)])
        T = max(map(len, rows))
        ids = torch.tensor([[0] * (T - len(r)) + r for r in rows])
        mask = torch.tensor([[0] * (T - len(r)) + [1] * len(r) for r in rows])
        return {"input_ids": ids, "attention_mask": mask}


def _tiny():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    cfg = Qwen3VLConfig(
        text_config=dict(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, pad_token_id=0,
                         rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3]}),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                           deepstack_visual_indexes=[0]))
    torch.manual_seed(0)
    return Qwen3VLForConditionalGeneration(cfg)


def test_answer_start():
    assert TL.answer_start([1, 2, 250, 251, 252, 9, 9], MARK) == 5
    assert TL.answer_start([250, 251, 252, 1, 250, 251, 252, 7], MARK) == 7      # the last one
    assert TL.answer_start([1, 2, 3], MARK) == 3


def test_train_lora_cpu(tmp_path, monkeypatch):
    from PIL import Image
    import transformers
    import peft
    M, pack, out, gold = _fake_trial(tmp_path, monkeypatch)
    frames = []
    for k in range(7):
        f = tmp_path / f"f{k}.jpg"
        Image.new("RGB", (16, 16), (k * 30, 0, 0)).save(f)
        frames.append(str(f))
    recs = [{"video_id": "v", "t": float(i), "frames": frames,
             "target": TL.lora_target(["close-up"] if i % 2 else ["wide", "dancing"], TL.parse_vocabulary(
                 "tag,group\nclose-up,shot\nwide,shot\ndancing,action\n"))} for i in range(40)]
    (out / "lora").mkdir(parents=True)
    (out / "lora" / "data.json").write_text(json.dumps({
        "created": "c1", "head": "h", "moment": "m", "tail": "t", "vocab_csv": "x", "test_videos": [],
        "val_videos": ["v2"], "train": recs[:32], "val": recs[32:]}))
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *a, **k: _Proc())
    monkeypatch.setattr(transformers.Qwen3VLForConditionalGeneration, "from_pretrained", lambda *a, **k: _tiny())
    monkeypatch.setattr(transformers, "BitsAndBytesConfig", lambda **k: None)
    monkeypatch.setattr(peft, "prepare_model_for_kbit_training", lambda m, **k: m)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a: type("P", (), {"total_memory": 80e9})())
    monkeypatch.setattr(M, "LORA_SAVE_EVERY", 16)

    # out of memory on the first 8-at-once pass → halves to 4 and carries on
    real = M._answer_loss
    calls = {"n": 0}

    def flaky(model, batch, starts):
        calls["n"] += 1
        if calls["n"] == 1 and batch["input_ids"].shape[0] == 8:
            raise torch.cuda.OutOfMemoryError("fake")
        return real(model, batch, starts)
    monkeypatch.setattr(M, "_answer_loss", flaky)

    # stop the first run right after its first checkpoint is saved (16 examples) to test resume
    vol = M.pack_vol

    class StopAfterCommit:
        def reload(self):
            vol.reload()

        def commit(self):
            vol.commit()
            raise KeyboardInterrupt("cut off")
    monkeypatch.setattr(M, "pack_vol", StopAfterCommit())
    with pytest.raises(KeyboardInterrupt):
        M.train_lora.local("t1", 1.0, 1e-3, 8, 0, True, 8)
    prog = json.loads((out / "lora" / "t1" / "checkpoint" / "progress.json").read_text())
    assert prog["n"] == 16 and prog["total"] == 32 and prog["seen"] == 16
    assert (out / "lora" / "t1" / "data.json").exists()

    monkeypatch.setattr(M, "pack_vol", vol)
    r = M.train_lora.local("t1", 1.0, 1e-3, 8, 0, False, 8)            # resume
    assert r["examples"] == 32 and r["micro_batch"] == 8 and r["val_loss"] is not None
    assert (out / "lora" / "t1" / "adapter_model.safetensors").exists()
    assert not (out / "lora" / "t1" / "checkpoint").exists()          # removed once finished
    meta = json.loads((out / "lora" / "t1" / "meta.json").read_text())
    assert meta["train_loss_end"] > 0


def test_labels_loss_matches_sliced_loss(tmp_path, monkeypatch):
    """The Unsloth path takes the loss from labels; it must equal the answer-only sliced loss."""
    M, *_ = _fake_trial(tmp_path, monkeypatch)
    m = _tiny().eval()
    r0, r1 = torch.randint(1, 200, (12,)), torch.randint(1, 200, (9,))
    ids = torch.stack([r0, torch.cat([torch.zeros(3, dtype=torch.long), r1])])
    mask = torch.stack([torch.ones(12, dtype=torch.long),
                        torch.cat([torch.zeros(3, dtype=torch.long), torch.ones(9, dtype=torch.long)])])
    with torch.no_grad():
        a, na = M._answer_loss(m, {"input_ids": ids, "attention_mask": mask}, [8, 9])
        b, nb = M._answer_loss_labels(m, {"input_ids": ids, "attention_mask": mask}, [8, 9])
    assert int(na) == int(nb) == 7
    assert abs(float(a) - float(b)) < 1e-3


def test_plain_target_modules(tmp_path, monkeypatch):
    M, *_ = _fake_trial(tmp_path, monkeypatch)
    f = tmp_path / "adapter_config.json"
    f.write_text(json.dumps({"r": 16, "target_modules": "(?:.*?language.*?(q_proj|v_proj).*?)"}))
    M._plain_target_modules(f)
    cfg = json.loads(f.read_text())
    assert cfg["target_modules"] == M.LORA_TARGET and cfg["r"] == 16
