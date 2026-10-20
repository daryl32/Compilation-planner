"""Media Library tidy-up: every library runs with its finder in the page, filters apply
straight away, and the render cards' Play / Details / ⋯ More actions work."""

from __future__ import annotations

import subprocess

import pytest

from tests.support import DATA  # noqa: F401  (installs the test config first)
from tests.apptest_util import errors, page_app

PAGE = "pages/3_Media_Library.py"


def _app(kind: str):
    at = page_app(PAGE, timeout=300)
    at.session_state["lib_kind"] = kind
    return at.run()


def test_videos_finder_and_status():
    at = _app("🎬 Videos")
    assert not errors(at), errors(at)
    assert any("of 6 videos" in m.value for m in at.markdown)
    assert not at.metric                                   # one status line, no big numbers
    at.text_input(key="lib_name_query").set_value("vid_a").run()
    assert any(m.value.startswith("**1** of 6 videos") for m in at.markdown)
    assert at.button(key="lib_toggle_vid_a").label == "⏱ Range"
    at.text_input(key="lib_name_query").set_value("").run()
    at.selectbox(key="lib_range_filter").set_value("Has a range").run()   # inside "More filters"
    assert not errors(at), errors(at)
    assert any(m.value.startswith("**0** of 6 videos") for m in at.markdown)   # no ranges in the test library
    assert not at.sidebar.text_input                       # filters left the sidebar


def test_audio_finder_and_actions():
    at = _app("🎵 Audio")
    assert not errors(at), errors(at)
    assert any("of 5 tracks" in m.value for m in at.markdown)
    labels = {b.label for b in at.button}
    assert {"▶️ Play", "🔬 Analysis", "🎭 Story map"} <= labels
    at.button(key="aud_detail_song_full").click().run()
    assert not errors(at), errors(at)
    assert at.button(key="aud_detail_song_full").label == "✕ Analysis"


@pytest.fixture()
def renders_dir(tmp_path):
    import config
    import renders
    mp = pytest.MonkeyPatch()
    d = tmp_path / "previews"
    d.mkdir()
    for n in ("song_full_2026-10-19_2010", "song_slow_2026-10-20_0915"):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=10",
                        "-f", "lavfi", "-i", "sine=frequency=440", "-t", "2", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(d / f"{n}.mp4")], check=True)
    mp.setattr(config, "PREVIEW_DIR", d)
    mp.setattr(renders, "PREVIEW_DIR", d)
    yield d
    mp.undo()


def test_renders_cards_and_tools(renders_dir):
    at = _app("🎞️ Renders")
    assert not errors(at), errors(at)
    assert any("of 2 renders" in m.value for m in at.markdown)
    plays = [b for b in at.button if (b.key or "").startswith("ren_play_")]
    assert len(plays) == 2 and plays[0].label == "▶️ Play"
    name = plays[0].key[len("ren_play_"):]
    at.button(key=f"ren_detail_{name}").click().run()        # ℹ️ Details opens rating / notes
    assert not errors(at), errors(at)
    assert any(r.label == "Rating" for r in at.radio)
    at.button(key=f"ren_pl_{name}").click().run()            # ⋯ More → 🎞️ Add to playlist
    assert at.session_state["ren_playlist"] == [name]
    assert any("Playlist" in m.value for m in at.markdown)   # the playlist panel shows once it has one
    at.button(key=f"ren_del_{name}").click().run()           # ⋯ More → 🗑️ Delete asks first
    assert any(b.key == f"ren_del_yes_{name}" for b in at.button)
    at.button(key=f"ren_del_no_{name}").click().run()
    assert not errors(at), errors(at)
