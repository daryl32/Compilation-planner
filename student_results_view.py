"""📊 Results and 🗄️ Teacher — inside 🎯 Training Labels → 🧪 Trial results (trial_view.py routes here).
Design: claude/results-pages-audit.md (9 Nov) — one Results page answering "is the new run better?" and "what
should I label next?", with 📥 a markdown copy for a Claude chat; the teacher's tools in an archive tab.
The numbers come from student_results.py (run_rows, headline, label_next, report_markdown)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import student_results as SRS
import training_labels as TLB
import trial_results as TR
from trial_logic import not_in_f1

RESULTS, TEACHER = "📊 Results", "🗄️ Teacher"


def _pct(x):
    return None if x is None else round(x * 100)


def _f1(x):
    return None if x is None else round(float(x), 3)


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


# ---------------------------------------------------------------------------
# 📊 Results
# ---------------------------------------------------------------------------

def _head_box(head: dict, best: str | None) -> None:
    last, prev = head["last"], head["prev"]
    what = "High F1" if head["key"] == "high" else "F1"
    with st.container(border=True):
        c = st.columns(4)
        c[0].metric(f"{what} · {last['name']}" + (" ⭐" if last["in_use"] else ""), _fmt(last[head["key"]]),
                    delta=None if head["delta"] is None else f"{head['delta']:+.2f} vs {prev['name']}",
                    help="High = position, action and camera focus — the groups the matcher leans on most.")
        c[1].metric("Medium F1", _fmt(last["medium"]), help="Clothing, prop and POV.")
        c[2].metric("All tags F1", _fmt(last["f1"]),
                    delta=None if last["vs_teacher"] is None else f"{last['vs_teacher']:+.2f} vs teacher",
                    help="Every tag except hair, shot size and text on screen.")
        c[3].metric("Test frames", last.get("frames") or "–", help="Your labelled frames in the 🧪 test videos.")
        icon = {"within chance": "➖"}.get(head["verdict"], "✅" if head["verdict"].startswith("better") else
                                           "⚠️" if head["verdict"].startswith("worse") else "🆕")
        st.markdown(f"{icon} **{last['name']}**"
                    + (f" is **{head['delta']:+.2f}** on {what} vs {prev['name']} — **{head['verdict']}** "
                       f"(chance: ±{head['noise']:.2f})" if prev and head["delta"] is not None else " — " + head["verdict"])
                    + (f" · 🏆 best so far: **{best}**" if best and best != last["name"] else ""))


def _fmt(x) -> str:
    return "–" if x is None else f"{x:.2f}"


def _runs_table(rows: list, baseline: dict | None, best: str | None, details: bool) -> None:
    out = []
    for r in reversed(rows):
        line = {"_id": f"student:{r['name']}", "Run": ("⭐ " if r["in_use"] else "") + ("🏆 " if r["name"] == best else "")
                + r["name"], "When": r["created"], "High F1": _f1(r["high"]), "Medium F1": _f1(r["medium"]),
                "F1": _f1(r["f1"])}
        if details:
            line.update({"vs teacher": r["vs_teacher"], "Right %": _pct(r["precision"]), "Found %": _pct(r["recall"]),
                         "Wrong": r["fp"], "Missed": r["fn"], "Frames": r["frames"], "Training F1": _f1(r["train_f1"]),
                         "Your labels": r["labels"], "Batch answers": r["partial"],
                         "F1 when trained": _f1(r["first_f1"]), "Min": r["minutes"], "Settings": r["settings"]})
        line["What changed"] = r["changed"]
        out.append(line)
    if baseline:
        b = {"_id": "teacher:baseline", "Run": baseline["name"], "When": None, "High F1": _f1(baseline["high"]),
             "Medium F1": _f1(baseline["medium"]), "F1": _f1(baseline["f1"])}
        if details:
            b.update({k: None for k in out[0] if k not in b and k != "What changed"})
        b["What changed"] = "Baseline — the teacher on the same frames"
        out.append(b)
    cfg = {c: _f1_col(c) for c in ("High F1", "Medium F1", "F1")}
    cfg.update({"What changed": st.column_config.TextColumn(width="large"),
                "Your note": st.column_config.TextColumn(width="medium"),
                "vs teacher": st.column_config.NumberColumn(format="%+.2f"),
                "Training F1": st.column_config.NumberColumn(format="%.2f"),
                "F1 when trained": st.column_config.NumberColumn(format="%.2f")})
    _notes_table(out, "srv_runs", cfg)


def _label_table(rows: list, label_run: str) -> list:
    groups = SRS.group_of_tags(TLB.load_vocabulary()[0])
    lr = SRS.label_next(SRS.load_detail(label_run), groups)
    if not lr:
        st.caption("No per-tag test scores for this run.")
        return lr
    df = pd.DataFrame([{"Group": f"{x['group']} · {x['importance']}", "Tag": x["tag"] + (" ⚠️" if x["worse_than_teacher"] else ""),
                        "Test F1": _f1(x["f1"]), "Errors": x["errors"], "Wrong": x["wrong"], "Missed": x["missed"],
                        "Teacher errors": x["teacher_errors"], "Training labels": x["train_labels"],
                        "Teacher trust": _f1(x["teacher_trust"])} for x in lr])
    st.dataframe(df, hide_index=True, width="stretch", key="srv_label_next",
                 column_config={"Test F1": _f1_col("Test F1"),
                                "Teacher trust": st.column_config.NumberColumn(format="%.2f",
                                    help="The teacher's F1 on your training labels — low means the student learns "
                                         "this tag mostly from your labels.")})
    st.caption("High groups first, most test errors first. ⚠️ = the student makes more than "
               f"{SRS.PER_TAG_NOISE} errors more than the teacher. Few **training labels** + low **teacher trust** = "
               "the tag your labels help most.")
    return lr


def _compare_runs(rows: list) -> None:
    if len(rows) < 2:
        return
    with st.expander("🔀 Compare two runs — per tag"):
        ids = [r["name"] for r in rows]
        a, b = st.columns(2)
        ia = a.selectbox("Before", ids, index=len(ids) - 2, key="srv_cmp_a")
        ib = b.selectbox("After", ids, index=len(ids) - 1, key="srv_cmp_b")
        pa = {t: x.get("test") for t, x in ((SRS.load_detail(ia) or {}).get("per_tag") or {}).items() if x.get("test")}
        pb = {t: x.get("test") for t, x in ((SRS.load_detail(ib) or {}).get("per_tag") or {}).items() if x.get("test")}
        ch = [c for c in TR.tag_changes({"per_tag": pa}, {"per_tag": pb}) if not not_in_f1(c["tag"])]
        if ch:
            st.dataframe([{"Tag": c["tag"], "Wrong": f"{c['wrong_a']} → {c['wrong_b']}",
                           "Missed": f"{c['missed_a']} → {c['missed_b']}",
                           "Errors": ("✅ " if c["change"] < 0 else "⚠️ ") + f"{c['change']:+d}"
                                     + (" (chance)" if abs(c["change"]) <= SRS.PER_TAG_NOISE else "")}
                          for c in ch], hide_index=True, width="stretch", key="srv_cmp_tags")
            changed = next((r["changed"] for r in rows if r["name"] == ib), "")
            if changed:
                st.caption(f"**{ib}** changed: {changed}")
        else:
            st.caption("No tag changed.")


def results(data: dict | None) -> None:
    import hair_vote as HV
    import library_common as LC
    idx = SRS.load_index()
    runs = (idx or {}).get("runs") or []
    groups = SRS.group_of_tags(TLB.load_vocabulary()[0])
    rows = SRS.run_rows(runs, groups)
    if not rows:
        _no_runs_note()
        return
    baseline = SRS.teacher_baseline(rows, groups)
    best = SRS.best_high(rows)
    head = SRS.headline(rows)
    _head_box(head, best)

    st.markdown("##### Runs")
    details = st.toggle("Show details", key="srv_details",
                        help="Right / Found %, wrong / missed, training F1, labels, settings and more.")
    _runs_table(rows, baseline, best, details)
    st.caption("Newest first · ⭐ in use · 🏆 best High F1 · **Your note** saves as you type. Runs are scored on "
               "today's test labels after `rescore-all`.")
    _chart([{"run": r["name"], "who": "High F1", "f1": r["high"]} for r in rows if r["high"] is not None]
           + [{"run": r["name"], "who": "All tags F1", "f1": r["f1"]} for r in rows], "Test F1")

    st.markdown("##### What to label next")
    names = [r["name"] for r in reversed(rows)]
    label_run = st.selectbox("Run", names, key="srv_label_run")
    lr = _label_table(rows, label_run)
    _compare_runs(rows)

    try:
        vids = TR.test_set_videos()
    except Exception:
        vids = set()
    hair = HV.test_check(vids, LC.video_hair()) if vids else None
    sh = rows[-1].get("shot") or {}
    bits = []
    if hair and hair.get("checked"):
        bits.append(f"💇 Hair right **{hair['right']} / {hair['checked']}** test videos")
    if sh.get("frames"):
        bits.append(f"🎥 Shot within one step **{_pct(sh.get('within_one'))}%** ({rows[-1]['name']})")
    if bits:
        st.markdown(" · ".join(bits))
    with st.expander("💇 Hair colour and 🎥 shot size — detail"):
        _hair_check()
        _shot_check(runs, [], best)

    st.download_button(
        "📥 Download results (.md)", key="srv_download_md", width="stretch",
        data=SRS.report_markdown(rows, baseline, head, lr, label_run, hair, idx.get("created") or ""),
        file_name=f"student-results-{rows[-1]['name']}.md", mime="text/markdown",
        help="The headline, every run, the per-tag table for the run picked above, hair and shot — paste it into a "
             "Claude chat to talk the results through.")


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



# ---------------------------------------------------------------------------
# 🗄️ Teacher — the archive (teacher tuning finished 3 Nov; useful again if more Modal batches are run)
# ---------------------------------------------------------------------------

def teacher_runs(data: dict | None) -> None:
    """Teacher test runs and the 10-video batches, with their notes, then the batch details."""
    import trial_view as TV
    scored = _teacher_tests()
    if scored:
        st.markdown("##### 🧪 Teacher test runs")
        rows = []
        for x in reversed(scored):
            o, run = x["score"]["overall"], x["run"]
            rows.append({"_id": f"test:{run['id']}", "Run": run.get("label") or run["id"],
                         "When": run.get("created") or run["id"][:16], "Frames": x["score"]["n"], "F1": _f1(o["f1"]),
                         "Right %": _pct(o["precision"]), "Found %": _pct(o["recall"]), "Wrong": o["fp"],
                         "Missed": o["fn"], "Settings": _teacher_test_settings(x), "What changed": run.get("note") or ""})
        _notes_table(rows, "srv_teacher_tests", {"F1": _f1_col(), "What changed": st.column_config.TextColumn(width="large")})
        TV._test_compare(scored)
    batches = [b for b in _teacher_batches(data) if b["batch"] not in ("All", "?")]
    if batches:
        st.markdown("##### 📦 Teacher batches")
        rows = []
        for b in reversed(batches):
            ex = {"on": "worked examples on", "off": "", "mixed": "worked examples mixed"}.get(b.get("examples"), "")
            rows.append({"_id": f"batch:{b['batch']}", "Run": f"Batch {b['batch']}", "When": b.get("run"),
                         "Your frames": b.get("labelled"), "F1": _f1(b.get("f1")), "Right %": _pct(b.get("precision")),
                         "Found %": _pct(b.get("recall")), "Moments": b.get("moments"), "Min": b.get("tag_min"),
                         "Settings": ex})
        _notes_table(rows, "srv_teacher_batches", {"F1": _f1_col()})
    if data:
        TV._batch_details(data)
