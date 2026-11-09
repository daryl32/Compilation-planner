"""📋 Summary (landing), 🏋️ Training results and 🧪 Test results — inside 🎯 Training Labels → 🧪 Trial
results (trial_view.py routes here). Teacher and student runs side by side in one table per page, with
what changed and your own note per run, then per tag. Plan: claude/student-results-plan.md.

Training and test are kept apart on purpose: Training = how a run learned on the training videos (your
labels there, each checked by a fold model that didn't train on that video); Test = the fair score on
the 🧪 test-set videos no model ever trains on, against the teacher on the same frames."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import student_results as SRS
import training_labels as TLB
import trial_results as TR
from trial_logic import VIDEO_LEVEL_TAGS, shot_step

TRAINING, TEST = "🏋️ Training results", "🧪 Test results"


def _pct(x):
    return None if x is None else round(x * 100)


def _f1(x):
    return None if x is None else round(float(x), 3)


def _go(page: str) -> None:
    st.session_state["tr_sub"] = page


def _no_runs_note() -> None:
    st.info("No student runs here yet. On the laptop: `student_tags.bat rescore-all` (or any train / score) "
            "writes them to the Drive catalogue folder — then press ☁️ Sync from Google Drive.")


def _f1_col(label="F1"):
    return st.column_config.ProgressColumn(label, min_value=0.0, max_value=1.0, format="%.2f")


def _teacher_batches(data: dict | None) -> list:
    if not data:
        return []
    teacher = (TR.shown_teachers(data.get("teachers") or {}) or [TR.MAIN_TEACHER])[0]
    return TR.batch_report(TR.with_saved_labels(data["videos"], TLB.load_labels()), teacher)


def _teacher_tests() -> list:
    data = TR.load_test_runs()
    if not data:
        return []
    rows, _ = TLB.load_vocabulary()
    return TR.test_run_scores(data["runs"], TLB.load_labels(), [r["tag"] for r in rows])


def _teacher_test_settings(x: dict) -> str:
    """A teacher test run's own details, short: LoRA, 2nd looks, needs-a-tag fills, frames copied, cost."""
    run = x["run"]
    bits = [f"LoRA {run['lora']}"] if run.get("lora") else []
    if run.get("looks"):
        bits.append(f"2nd looks {run['looks']} (+{run.get('look_added', 0)}/−{run.get('look_removed', 0)})")
    if run.get("required_fills"):
        bits.append(f"needs-a-tag fills {run['required_fills']}")
    if x.get("copied"):
        bits.append(f"your frames copied {x['copied']}")
    if run.get("cost") is not None:
        bits.append(f"${run['cost']:.2f}")
    return " · ".join(bits)


def _marks(r: dict, best: str | None) -> str:
    return ("⭐ " if r.get("in_use") else "") + ("🏆 " if r["name"] == best else "")


def _notes_table(rows: list, key: str, column_config: dict) -> None:
    """The runs table with the Your note column editable; changed notes are saved at once (and queued
    for ☁️ Save edits to Drive)."""
    notes = SRS.load_notes()
    df = pd.DataFrame([{**{k: v for k, v in r.items() if k != "_id"}, "Your note": notes.get(r["_id"], "")}
                       for r in rows])
    ids = [r["_id"] for r in rows]
    edited = st.data_editor(df, hide_index=True, width="stretch", key=key, column_config=column_config,
                            disabled=[c for c in df.columns if c != "Your note"])
    clean = lambda n: "" if n is None or (isinstance(n, float) and pd.isna(n)) else str(n)
    changed = {i: clean(n) for i, n in zip(ids, edited["Your note"].tolist()) if clean(n) != notes.get(i, "")}
    if changed:
        SRS.save_notes({**notes, **changed})
        st.toast("📝 Note saved — ☁️ Save edits to Drive (sidebar) to keep it on Drive.")


def _chart(points: list, y_title: str) -> None:
    """F1 per run in the order they were trained (student line, teacher line where known)."""
    if len(points) < 2:
        return
    import altair as alt
    order = [p["run"] for p in points if p["who"] == "Student"]
    df = pd.DataFrame(points)
    ch = (alt.Chart(df).mark_line(point=True)
          .encode(x=alt.X("run:N", sort=order, title=None), y=alt.Y("f1:Q", title=y_title, scale=alt.Scale(zero=False)),
                  color=alt.Color("who:N", title=None), tooltip=["run", "who", alt.Tooltip("f1:Q", format=".3f")])
          .properties(height=220))
    st.altair_chart(ch, width="stretch")


def _grid(runs: list, side: str, key: str) -> None:
    """Tags × runs: F1 per tag per run, shaded, so tags that keep moving stand out."""
    details = {r["name"]: SRS.load_detail(r["name"]) for r in runs}
    tags, by = SRS.tag_grid(details, side)
    names = [r["name"] for r in runs if r["name"] in by]
    if not names:
        st.caption("No per-tag scores for these runs yet.")
        return
    df = pd.DataFrame({n: [by[n].get(t) for t in tags] for n in names}, index=tags)
    df["spread"] = df.max(axis=1) - df.min(axis=1)
    df = df.sort_values("spread", ascending=False)

    def shade(v):
        if v is None or pd.isna(v):
            return ""
        return f"background-color: rgba({int(229 - 183 * v)}, {int(83 + 77 * v)}, {int(75 - 8 * v)}, .28)"
    sty = df.style.format("{:.2f}", na_rep="–").map(shade, subset=names)
    st.dataframe(sty, width="stretch", key=key, height=min(38 + 35 * len(df), 640))
    st.caption("Sorted by how much a tag moves between runs (**spread**, highest F1 − lowest). Red → green = "
               "F1 0 → 1. A tag with few of your labels moves a lot by chance.")


# ---------------------------------------------------------------------------
# 📋 Summary — the landing page
# ---------------------------------------------------------------------------

def landing(data: dict | None) -> None:
    idx = SRS.load_index()
    runs = (idx or {}).get("runs") or []
    best = SRS.best_on_test(runs)
    last = runs[-1] if runs else None
    a, b = st.columns(2)
    with a.container(border=True):
        st.markdown(f"#### {TRAINING}")
        st.caption("What went into each run and how well it learned on the **training** videos — your labels "
                   "there, each checked by a fold model that never trained on that video. Plus the teacher's "
                   "batches.")
        c1, c2 = st.columns(2)
        tr = (last or {}).get("training") or {}
        c1.metric(f"Latest run {last['name']}" if last else "Latest run", "–" if tr.get("f1") is None else f"{tr['f1']:.2f}",
                  help="F1 on your labels in the training videos (runs from v12 on).")
        c2.metric("Your labels", f"{last.get('labels'):,}" if last and last.get("labels") else "–")
        batches = [r for r in _teacher_batches(data) if r["batch"] not in ("All", "?")]
        if batches:
            lb = batches[-1]
            st.caption(f"Teacher, batch {lb['batch']}: F1 {lb.get('f1') or 0:.2f} on {lb['labelled']} of your frames.")
        st.button(f"Open {TRAINING}", key="tr_go_train", on_click=_go, args=(TRAINING,), width="stretch")
    with b.container(border=True):
        st.markdown(f"#### {TEST}")
        st.caption("The fair score: the 🧪 test-set videos no model trains on, every run on **today's** labels, "
                   "next to the teacher on the same frames.")
        c1, c2 = st.columns(2)
        br = next((r for r in runs if r["name"] == best), None)
        t = (br or {}).get("test") or {}
        tf, sf = (t.get("teacher") or {}).get("f1"), (t.get("student") or {}).get("f1")
        c1.metric(f"Best student {best}" if best else "Best student", "–" if sf is None else f"{sf:.2f}",
                  delta=None if sf is None or tf is None else f"{sf - tf:+.2f} vs teacher", delta_color="normal")
        c2.metric("Test frames", t.get("frames") or "–", help="Your labelled test-set frames the teacher run also "
                  "tagged — every run is scored on these.")
        if br:
            st.caption(f"Noise at {t.get('frames')} frames: ±{SRS.noise_f1(t.get('frames')):.2f} F1. "
                       f"Scored {t.get('scored_at') or '?'}.")
        st.button(f"Open {TEST}", key="tr_go_test", on_click=_go, args=(TEST,), width="stretch")
    if not runs:
        _no_runs_note()
    elif idx.get("in_use"):
        st.caption(f"⭐ **{idx['in_use']}** is the model in use (its scores are in student_scores, which the new "
                   f"matcher will read) · 🏆 best on the test set: **{best or '–'}** · records written "
                   f"{idx.get('created')}.")


# ---------------------------------------------------------------------------
# 🏋️ Training results
# ---------------------------------------------------------------------------

def training(data: dict | None) -> None:
    import trial_view as TV
    idx = SRS.load_index()
    runs = (idx or {}).get("runs") or []
    best = SRS.best_on_test(runs)
    st.markdown(f"#### {TRAINING}")
    st.caption("**Student runs:** F1 / Right / Found on **your labels in the training videos**, each frame "
               "answered by the fold model that didn't train on its video (runs from v12 on — older runs show "
               "their data and what changed only). **Teacher batches:** the teacher against your labels per "
               "10-video batch. Not comparable with 🧪 Test results — different videos, different purpose.")
    _chart([{"run": r["name"], "who": "Student", "f1": r["training"]["f1"]} for r in runs
            if (r.get("training") or {}).get("f1") is not None], "F1 on your training labels")
    rows = []
    for r in reversed(runs):
        tr = r.get("training") or {}
        rows.append({"_id": f"student:{r['name']}", "Run": _marks(r, best) + r["name"], "Who": "Student",
                     "When": r.get("created"), "Your frames": tr.get("frames"), "F1": _f1(tr.get("f1")),
                     "Right %": _pct(tr.get("precision")), "Found %": _pct(tr.get("recall")),
                     "Your labels": r.get("labels"), "Teacher moments": r.get("teacher_moments"),
                     "Scene moments": r.get("scene_moments"), "Min": r.get("minutes"),
                     "Settings": SRS.settings_text(r.get("settings")), "What changed": r.get("changed")})
    for b in reversed(_teacher_batches(data)):
        if b["batch"] in ("All", "?"):
            continue
        ex = {"on": "worked examples on", "off": "", "mixed": "worked examples mixed"}.get(b.get("examples"), "")
        rows.append({"_id": f"batch:{b['batch']}", "Run": f"Batch {b['batch']}", "Who": f"Teacher {b.get('teacher') or ''}".strip(),
                     "When": b.get("run"), "Your frames": b.get("labelled"), "F1": _f1(b.get("f1")),
                     "Right %": _pct(b.get("precision")), "Found %": _pct(b.get("recall")), "Your labels": None,
                     "Teacher moments": b.get("moments"), "Scene moments": None, "Min": b.get("tag_min"),
                     "Settings": ex, "What changed": ""})
    if not rows:
        _no_runs_note()
        return
    _notes_table(rows, "srv_train_runs", {"F1": _f1_col(), "What changed": st.column_config.TextColumn(width="large"),
                                          "Your note": st.column_config.TextColumn(width="medium")})
    st.caption("⭐ in use · 🏆 best on the test set · **Your note** is yours to edit (saved at once; ☁️ Save edits "
               "to Drive keeps it). Newest first.")

    trained = [r for r in runs if (r.get("training") or {}).get("f1") is not None] or runs
    st.markdown("##### Per tag")
    name = st.selectbox("Run", [r["name"] for r in reversed(trained)], key="srv_train_tag_run")
    d = SRS.load_detail(name) or {}
    per = []
    for t, x in (d.get("per_tag") or {}).items():
        s = x.get("train") or {}
        per.append({"Tag": t, "Your labels": x.get("train_labels"), "F1": _f1(s.get("f1")),
                    "Wrong": s.get("fp"), "Missed": s.get("fn"),
                    "Teacher F1 on your labels": _f1(x.get("teacher_trust")),
                    "Teacher answers count for": _f1(x.get("trust_weight")), "Cut-off": x.get("cutoff")})
    if per:
        per.sort(key=lambda p: (-(p["Your labels"] or 0), p["Tag"].lower()))
        st.dataframe(per, hide_index=True, width="stretch", key="srv_train_tags",
                     column_config={"F1": _f1_col(), "Cut-off": st.column_config.NumberColumn(format="%.2f")})
        if all(p["F1"] is None for p in per):
            st.caption("This run is from before training-side scores were kept — retrain to see its F1 per tag. "
                       "Teacher trust and cut-offs are its own.")
    with st.expander("▦ Tags × runs — F1 on your training labels"):
        _grid([r for r in runs if (r.get("training") or {}).get("f1") is not None], "train", "srv_train_grid")
    if data:
        TV._batch_details(data)


# ---------------------------------------------------------------------------
# 🧪 Test results
# ---------------------------------------------------------------------------

def _hair_check() -> None:
    """💇 The model's hair vote vs your confirmed answers on the 🧪 test videos (claude/hair-colour-plan.md)."""
    import hair_vote as HV
    import library_common as LC
    import trial_results as TRR
    try:
        vids = TRR.test_set_videos()
    except Exception:
        vids = set()
    if not vids:
        return
    c = HV.test_check(vids, LC.video_hair())
    with st.container(border=True):
        if not c["confirmed"]:
            st.markdown("**💇 Hair colour per video** — confirm the test videos' hair in the Reviewer "
                        "(🧪 In the test set + 💇 Hair not confirmed) to check the vote here.")
            return
        if not c["checked"]:
            st.markdown(f"**💇 Hair colour per video** — {c['confirmed']} test videos confirmed, but no vote yet "
                        "(Reviewer → 🤖 Vote hair).")
            return
        m = st.columns(3)
        m[0].metric("💇 Hair right", f"{c['right']} / {c['checked']}",
                    help="Test videos where the vote's colours are exactly yours (a ❓ unsure vote isn't counted "
                         "as right).")
        m[1].metric("Main colour right", f"{c['main_right']} / {c['checked']}",
                    help="The vote's top colour is one of yours — even if it missed or added a second actress.")
        m[2].metric("❓ Unsure", c["unsure"], help="Votes that sent the video to you instead of guessing.")
        bits = [f"Vote from **{', '.join(c['models'])}**", f"{c['confirmed']} of {c['videos']} test videos confirmed"]
        if c["no_vote"]:
            bits.append(f"{c['no_vote']} confirmed without a vote (no student scores)")
        st.caption(" · ".join(bits) + ".")
        if c["wrong"]:
            with st.expander(f"Where the vote differs ({len(c['wrong'])})", expanded=False):
                st.dataframe(pd.DataFrame([{"Video": w["video_id"], "Vote": LC.hair_label(w["vote"]) + (" ❓" if w["unsure"] else ""),
                                            "Yours": LC.hair_label(w["yours"])} for w in c["wrong"]]),
                             hide_index=True, width="stretch", key="srv_hair_wrong")


def _shot_check(runs: list, scored: list, best: str | None) -> None:
    """🎥 Shot size on its scale (extreme close-up → close-up → medium → wide): the matcher only cares how
    far apart two shots are, so within one step is the number that counts."""
    rows = []
    for r in reversed(runs):
        sh = (r.get("test") or {}).get("shot")
        if sh and sh.get("frames"):
            rows.append({"Run": _marks(r, best) + r["name"], "Who": "Student", "Frames": sh["frames"],
                         "Within one step": _pct(sh.get("within_one")), "Exact": _pct(sh.get("exact")),
                         "Big misses": _pct(sh.get("big_miss")), "No shot given": _pct(sh.get("none_given"))})
    for x in reversed(scored):
        sh = x["score"].get("shot") or {}
        if sh.get("frames"):
            run = x["run"]
            rows.append({"Run": run.get("label") or run["id"], "Who": "Teacher", "Frames": sh["frames"],
                         "Within one step": _pct(sh.get("within_one")), "Exact": _pct(sh.get("exact")),
                         "Big misses": _pct(sh.get("big_miss")), "No shot given": _pct(sh.get("none_given"))})
    with st.container(border=True):
        st.markdown("**🎥 Shot size** — on the scale extreme close-up → close-up → medium → wide")
        if not rows:
            st.caption("No shot scores yet — run `deploy/student.sh rescore-all` on the server.")
            return
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", key="srv_shot_table")
        st.caption("**Within one step** is what the matcher needs (close-up next to medium is fine). **Big "
                   "misses** = 2+ steps, e.g. close-up ↔ wide — those are worth relabelling. Frames = your "
                   "labelled test frames with exactly one shot tag.")


def test() -> None:
    import trial_view as TV
    idx = SRS.load_index()
    runs = (idx or {}).get("runs") or []
    best = SRS.best_on_test(runs)
    scored = _teacher_tests()
    st.markdown(f"#### {TEST}")
    frames = next(((r.get("test") or {}).get("frames") for r in reversed(runs) if r.get("test")), None)
    st.caption("Your labelled 🧪 test-set frames — videos no model trains on. **Student runs** are scored on "
               "today's labels (`student_tags.bat rescore-all` after a test-labelling session), on the frames "
               "the reference teacher run also tagged, so every row is the same frames. **Vs teacher** = the "
               "student minus that teacher run on them. **F1 at the time** = its score when it was trained."
               + (f" At {frames} frames a difference under ±{SRS.noise_f1(frames):.2f} F1 is chance." if frames else "")
               + " 💇 Hair colour and 🎥 shot size are left out of every F1 — each has its own check below.")
    _hair_check()
    _shot_check(runs, scored, best)
    pts = []
    for r in runs:
        t = r.get("test") or {}
        if (t.get("student") or {}).get("f1") is not None:
            pts.append({"run": r["name"], "who": "Student", "f1": t["student"]["f1"]})
            if (t.get("teacher") or {}).get("f1") is not None:
                pts.append({"run": r["name"], "who": f"Teacher ({t.get('reference') or 'reference'})",
                            "f1": t["teacher"]["f1"]})
    _chart(pts, "F1 on your test labels")
    rows = []
    for r in reversed(runs):
        t = r.get("test") or {}
        s, te = t.get("student") or {}, t.get("teacher") or {}
        vs = None if s.get("f1") is None or te.get("f1") is None else round(s["f1"] - te["f1"], 3)
        rows.append({"_id": f"student:{r['name']}", "Run": _marks(r, best) + r["name"], "Who": "Student",
                     "When": r.get("created"), "Frames": t.get("frames"), "F1": _f1(s.get("f1")),
                     "Right %": _pct(s.get("precision")), "Found %": _pct(s.get("recall")), "Wrong": s.get("fp"),
                     "Missed": s.get("fn"), "Vs teacher": vs, "F1 at the time": _f1(t.get("first_f1")),
                     "Settings": SRS.settings_text(r.get("settings")), "What changed": r.get("changed")})
    for x in reversed(scored):
        o, run = x["score"]["overall"], x["run"]
        rows.append({"_id": f"test:{run['id']}", "Run": run.get("label") or run["id"], "Who": "Teacher",
                     "When": run.get("created") or run["id"][:16], "Frames": x["score"]["n"], "F1": _f1(o["f1"]),
                     "Right %": _pct(o["precision"]), "Found %": _pct(o["recall"]), "Wrong": o["fp"], "Missed": o["fn"],
                     "Vs teacher": None, "F1 at the time": None,
                     "Settings": _teacher_test_settings(x), "What changed": run.get("note") or ""})
    if not rows:
        _no_runs_note()
        return
    _notes_table(rows, "srv_test_runs", {"F1": _f1_col(), "F1 at the time": st.column_config.NumberColumn(format="%.2f"),
                                         "Vs teacher": st.column_config.NumberColumn(format="%+.2f"),
                                         "What changed": st.column_config.TextColumn(width="large"),
                                         "Your note": st.column_config.TextColumn(width="medium")})
    st.caption("⭐ in use · 🏆 best on the test set · teacher rows: each teacher test run on all your labelled "
               "frames it tagged (its own frame count). Newest first.")

    studs = [r for r in runs if r.get("test")]
    if studs:
        st.markdown("##### Per tag — student vs teacher")
        name = st.selectbox("Run", [r["name"] for r in reversed(studs)], key="srv_test_tag_run",
                            index=[r["name"] for r in reversed(studs)].index(best) if best else 0)
        d = SRS.load_detail(name) or {}
        per = []
        for t, x in (d.get("per_tag") or {}).items():
            s, te = x.get("test") or {}, x.get("test_teacher") or {}
            n = (s.get("tp") or 0) + (s.get("fn") or 0)
            es, et = SRS.errors(s), SRS.errors(te) if te else None
            if not n and not es and not et:
                continue
            diff = None if es is None or et is None else es - et
            if t.lower() in {v.lower() for v in VIDEO_LEVEL_TAGS}:
                t = f"{t} 💇 (per video — not in F1)"
            elif shot_step(t) is not None:
                t = f"{t} 🎥 (shot scale — not in F1)"
            per.append({"Tag": t, "Your labels": n, "Student F1": _f1(s.get("f1")), "Student errors": es,
                        "Teacher F1": _f1(te.get("f1")) if te else None, "Teacher errors": et, "Difference": diff})
        per.sort(key=lambda p: (-(p["Difference"] if p["Difference"] is not None else -999), p["Tag"].lower()))
        if per:
            df = pd.DataFrame(per)

            def grey(row):
                small = row["Difference"] is not None and abs(row["Difference"]) <= SRS.PER_TAG_NOISE
                return ["color: rgba(128,128,128,.6)" if small else ""] * len(row)
            st.dataframe(df.style.apply(grey, axis=1).format({"Student F1": "{:.2f}", "Teacher F1": "{:.2f}"},
                                                             na_rep="–"),
                         hide_index=True, width="stretch", key="srv_test_tags")
            st.caption(f"Errors = wrong + missed. Most extra student errors first. Greyed: within "
                       f"±{SRS.PER_TAG_NOISE} errors of the teacher — chance at this size.")
        if len(studs) >= 2:
            with st.expander("🔀 Compare two student runs — per tag", expanded=False):
                ids = [r["name"] for r in studs]
                a, b = st.columns(2)
                ia = a.selectbox("Before", ids, index=len(ids) - 2, key="srv_cmp_a")
                ib = b.selectbox("After", ids, index=len(ids) - 1, key="srv_cmp_b")
                pa = {t: x.get("test") for t, x in ((SRS.load_detail(ia) or {}).get("per_tag") or {}).items() if x.get("test")}
                pb = {t: x.get("test") for t, x in ((SRS.load_detail(ib) or {}).get("per_tag") or {}).items() if x.get("test")}
                ch = TR.tag_changes({"per_tag": pa}, {"per_tag": pb})
                if ch:
                    st.dataframe([{"Tag": c["tag"], "Wrong": f"{c['wrong_a']} → {c['wrong_b']}",
                                   "Missed": f"{c['missed_a']} → {c['missed_b']}",
                                   "Errors": ("✅ " if c["change"] < 0 else "⚠️ ") + f"{c['change']:+d}"
                                             + (" (chance)" if abs(c["change"]) <= SRS.PER_TAG_NOISE else "")}
                                  for c in ch], hide_index=True, width="stretch", key="srv_cmp_tags")
                    st.caption(f"**{ib}** changed: " + next(r for r in studs if r["name"] == ib)["changed"])
                else:
                    st.caption("No tag changed.")
        with st.expander("▦ Tags × runs — F1 on your test labels"):
            _grid(studs, "test", "srv_test_grid")
    if scored:
        TV._test_compare(scored)
