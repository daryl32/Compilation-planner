"""siglip_tune.py — the SigLIP2 fine-tune experiment, end to end on a made-up library with a tiny
random SigLIP (CPU): pick the moments, cut the pictures, train, embed again, train the student on them."""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest

import siglip_tune as FT
import student_tags as S
import trial_logic as TL
from tests.test_student_tags import VOCAB_CSV, _library


def test_pick_rows_keeps_your_answers_and_caps_the_rest(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "WORK", tmp_path)
    labels, test = _library(tmp_path, n_videos=6)
    vocab = TL.parse_vocabulary(VOCAB_CSV)
    ds = S.build_dataset(vocab, labels, set(test), root=tmp_path / "videos", scenes={})
    T = FT.embedding_times(ds, tmp_path / "videos")
    sel = FT.pick_rows(ds, set(test), T, per_video=5, max_frames=10_000)
    assert not set(sel["video"]) & set(test)                       # the test set is never used
    mine = ds["is_label"][sel["row"]]
    learn = ~np.isin(ds["video"], test)
    assert mine.sum() == (ds["is_label"] & learn).sum() == sel["mine"]      # every label of yours
    for v in set(sel["video"]):
        assert (~mine[sel["video"] == v]).sum() <= 5               # the teacher's: at most 5 a video
    assert len({(v, f) for v, f in zip(sel["video"], sel["frame"])}) == len(sel["row"])   # one per picture
    # the picture is the 15-fps frame of the embedding nearest the moment
    r = sel["row"][0]
    assert sel["frame"][0] == FT.frame_index(T[ds["idx"][r, S.HALF]])
    again = FT.pick_rows(ds, set(test), T, per_video=5, max_frames=10_000)
    assert (again["row"] == sel["row"]).all()                      # the same pick every time
    small = FT.pick_rows(ds, set(test), T, per_video=5, max_frames=sel["mine"] + 3)
    assert len(small["row"]) == sel["mine"] + 3


def test_emb_file_and_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "WORK", tmp_path)
    _library(tmp_path, n_videos=3)
    assert S.emb_file("") == "embeddings.npz" and S.emb_file("ft1") == "embeddings_ft1.npz"
    assert S.emb_missing("") == [] and len(S.emb_missing("ft1")) == 3
    shutil.copy(tmp_path / "videos" / "vid00" / "embeddings.npz", tmp_path / "videos" / "vid00" / "embeddings_ft1.npz")
    assert S.emb_missing("ft1") == ["vid01", "vid02"]
    monkeypatch.setattr(S, "EMB_NAME", "ft1")
    with pytest.raises(SystemExit):                                # never a silent mix of old and new
        S._prepare(S.LABEL_WEIGHT)


def _tiny_encoder(half: bool):
    import torch
    from transformers import SiglipConfig, SiglipModel

    torch.manual_seed(0)
    cfg = SiglipConfig(vision_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 4,
                                      "num_attention_heads": 2, "image_size": 32, "patch_size": 16},
                       text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
                                    "num_attention_heads": 2, "vocab_size": 50, "max_position_embeddings": 8})
    model = SiglipModel(cfg)
    model.text_model = None

    def proc(images, return_tensors="pt"):
        import cv2
        px = np.stack([cv2.resize(im, (32, 32)).astype(np.float32) / 127.5 - 1 for im in images])
        return {"pixel_values": torch.as_tensor(px).permute(0, 3, 1, 2)}

    return model, proc


def test_end_to_end_on_a_tiny_encoder(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("cv2")
    pytest.importorskip("modal")             # modal_trial's decode helpers (nothing runs on Modal)
    if not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg")
    monkeypatch.setattr(S, "WORK", tmp_path)
    monkeypatch.setattr(S, "HERE", tmp_path)
    monkeypatch.setattr(S, "EMB_NAME", "")       # put back afterwards: the commands below change it
    _library(tmp_path, n_videos=8, minutes=0.5)
    clip = tmp_path / "clip.mp4"                 # 28 s, so the last embeddings fall past its end
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=28:size=320x180:rate=25",
                    "-pix_fmt", "yuv420p", str(clip)], check=True)
    monkeypatch.setattr(FT, "_proxy", lambda v: (clip, {"proxy": {"width": 320, "height": 180}}))
    monkeypatch.setattr(FT, "load_encoder", _tiny_encoder)

    assert FT.main(["check", "--blocks", "2", "--batch", "4", "--per-video", "4"]) == 0    # runs before anything exists
    assert FT.main(["frames", "--per-video", "4"]) == 0
    cut = list((tmp_path / "ft_frames").rglob("*.jpg"))
    assert cut and not list((tmp_path / "ft_frames").rglob("*.tmp.jpg"))
    assert FT.main(["frames", "--per-video", "4"]) == 0            # nothing new to cut
    assert len(list((tmp_path / "ft_frames").rglob("*.jpg"))) == len(cut)

    assert FT.main(["train", "--name", "t", "--blocks", "2", "--epochs", "2", "--batch", "8",
                    "--per-video", "4"]) == 0
    enc = tmp_path / "encoders" / "t"
    meta = json.loads((enc / "meta.json").read_text())
    assert meta["blocks"] == 2 and len(meta["history"]) == 2 and meta["best_epoch"] in (1, 2)
    import torch
    saved = torch.load(enc / "tuned.pt", map_location="cpu")["encoder"]
    assert set(saved) == set(meta["weights"])
    assert all(".layers.0." not in n and ".layers.1." not in n for n in saved)      # the lower blocks stay as they were
    assert any(".layers.3." in n for n in saved) and any("head" in n for n in saved)
    # the tuned weights really differ from the untouched model's, and load back in
    base, _ = _tiny_encoder(False)
    sd = base.state_dict()
    assert any(not torch.allclose(sd[n], t.float(), atol=1e-4) for n, t in saved.items())
    FT.apply_tuned(base, saved)
    with pytest.raises(SystemExit):
        FT.apply_tuned(base, {"not.a.weight": torch.zeros(1)})

    assert FT.main(["embed", "--name", "t", "--only", "vid00"]) == 0
    assert S.emb_missing("t") and FT.main(["embed", "--name", "t"]) == 0
    assert S.emb_missing("t") == []
    with np.load(tmp_path / "videos" / "vid03" / "embeddings.npz") as a, \
            np.load(tmp_path / "videos" / "vid03" / "embeddings_t.npz") as b:
        assert (a["frame_times"] == b["frame_times"]).all()         # same moments, same order
        e = b["frame_embs"].astype(np.float32)
        assert e.shape == (len(a["frame_times"]), 32) and e.dtype == np.float32
        assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-2)
        assert (e[-1] == e[-2]).all()                               # past the clip's end: the neighbour's
    # the untouched encoder through the same path gives something else (the weights are in use)
    m0, p0 = FT.load_tuned("")
    plain, _ = FT.embed_video(m0, p0, clip, {"proxy": {"width": 320, "height": 180}}, np.array([0.0, 0.53, 1.07]))
    assert not np.allclose(plain.astype(np.float32), e[:3], atol=1e-3)

    # the student trains on them, records it, and scores again from the saved model
    assert S.main(["train", "--name", "s-t", "--epochs", "3", "--folds", "2", "--emb", "t"]) == 0
    smeta = json.loads((tmp_path / "models" / "s-t" / "meta.json").read_text())
    assert smeta["emb"] == "t" and smeta["settings"]["emb"] == "t" and smeta["dim"] == 32
    S.use_emb("")
    assert S.main(["score", "--name", "s-t"]) == 0 and S.EMB_NAME == "t"
    assert S.main(["train", "--name", "s-0", "--epochs", "3", "--folds", "2"]) == 0
    assert json.loads((tmp_path / "models" / "s-0" / "meta.json").read_text())["settings"]["emb"] == "original"
    assert S.main(["rescore-all"]) == 0 and S.EMB_NAME == ""
    import student_runs as SR
    a = json.loads((tmp_path / "models" / "s-0" / "meta.json").read_text())
    b = json.loads((tmp_path / "models" / "s-t" / "meta.json").read_text())
    assert "picture embeddings t (was original)" in SR.what_changed(a, b)
    # on a machine without the tuned embeddings, rescore-all leaves that run alone
    for f in (tmp_path / "videos").rglob("embeddings_t.npz"):
        f.rename(f.with_name("gone.npz"))
    assert S.main(["rescore-all"]) == 0
    assert S.main(["export", "--name", "s-t"]) == 1
