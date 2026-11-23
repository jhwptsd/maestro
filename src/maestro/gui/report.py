# The report page: a player with live charts and the problem areas (rewritten).
#
# Input: the Performance filled by maestro.analyze(). gui/player.py turns it into JSON data, gui/player.html is the
# page (video/audio with the pose skeleton beside it, charts that trace out as it plays, a timeline of problems, the
# worst bar of each measure, a sortable event table and, folded away, the side-by-side score viewer of gui/viewer.py).
# `build_page` returns the complete HTML; the media is referenced by URL, so gui/server.py serves it with range
# requests (needed for seeking) and scripts/analyze.py saves it next to report.html (`copy_media`).
import json
import shutil
from pathlib import Path

from maestro.feedback.prompts import build_digest
from maestro.gui.player import player_data
from maestro.gui.viewer import build_viewer

TEMPLATE = (Path(__file__).parent / "player.html").read_text(encoding="utf-8")

# Styles of the score viewer (gui/viewer.py + gui/notation.py markup).
CSS = """
.viewer{display:grid;grid-template-columns:1fr 1fr;gap:10px}.vcol{min-width:0;border:1px solid #e4e4e4;background:white}
.vlegend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:.85em;color:#444;margin:.6em 0}
.vlegend i{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:5px;vertical-align:-1px}
.vsummary{font-size:.9em;color:#444}
.note.hl{filter:drop-shadow(0 0 3px #0a64ff)}
"""

# Hover a note in either panel of the score viewer to highlight its partner in the other. Scripts inside HTML injected
# with innerHTML do not run, so the page loads this once and it listens on the document.
VIEWER_JS = """
document.addEventListener("mouseover", function (e) {
  document.querySelectorAll(".note.hl").forEach(function (n) { n.classList.remove("hl"); });
  var t = e.target.closest && e.target.closest(".note[data-pair]");
  if (!t || !t.dataset.pair) return;
  document.querySelectorAll('.note[data-pair="' + t.dataset.pair + '"]').forEach(function (n) { n.classList.add("hl"); });
});
"""


def media_source(perf):
    """(path, "video" | "audio") of what the page plays: the converted video, or the upload when it has no picture."""
    return (perf.video, "video") if perf.video else (perf.audio_source, "audio")


def copy_media(perf, directory):
    """Make sure the media sits in `directory` and return its file name (for build_page's relative URL)."""
    path, _ = media_source(perf)
    directory = Path(directory)
    if Path(path).resolve().parent != directory.resolve():
        shutil.copy(path, directory / Path(path).name)
    return Path(path).name


def build_page(perf, media_url, run=None):
    """Complete HTML page for a Performance returned by analyze(); `media_url` is where the browser finds the recording.
    `run` (given by gui/server.py) lets the page ask the server for written feedback (feedback/generator.py)."""
    _, kind = media_source(perf)
    data = json.dumps(player_data(perf, media_url, kind, run, build_digest(perf))).replace("</", "<\\/")
    viewer = f"<details><summary>Score as read vs score as played</summary>{build_viewer(perf)}</details>"
    return (TEMPLATE.replace("/*CSS*/", CSS).replace("/*VIEWER_JS*/", VIEWER_JS)
            .replace("/*VIEWER*/", viewer).replace("/*DATA*/", data))
