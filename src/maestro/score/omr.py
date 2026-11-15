# Optical Music Recognition (OMR) module for processing and analyzing sheet music images.
#
# Only used when the score is an image/PDF: converts it to MusicXML, which then continues
# through preprocessing/score.py -> score/musicxml.py. Called by utils/convert_file_type.py, which has
# already sanity-checked PDFs (not encrypted/truncated/oversized).
# Use Audiveris and Partitura
#
# OMR is delegated to the external Audiveris program (https://github.com/Audiveris/audiveris),
# which must be installed and on PATH as `audiveris`. Audiveris reads multi-page PDFs directly. Quality
# depends on scan quality; always eyeball the result (see score/visualize.py).
import os
import shutil
import subprocess
from pathlib import Path


def _find_audiveris():
    """`audiveris` on PATH, else the default Windows install location (the MSI doesn't add it to PATH)."""
    exe = shutil.which("audiveris")
    if exe:
        return exe
    for root in (os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
        if root:
            candidate = Path(root) / "Audiveris" / "Audiveris.exe"
            if candidate.exists():
                return str(candidate)
    return None

from maestro.utils.errors import InputError, OMRUnavailableError

OMR_TIMEOUT = 600  # seconds; OMR of a long scan can be slow, but a hang should not block the app


def run_omr(path, out_dir, timeout=OMR_TIMEOUT):
    """Run Audiveris on an image/PDF -> path of the exported MusicXML (.mxl) file.

    Raises OMRUnavailableError if Audiveris is missing, InputError if it fails, times out, or finds
    no music (blank page, not a score). If the PDF holds several movements Audiveris writes one .mxl
    each; the largest (most music) is returned.
    """
    exe = _find_audiveris()
    if exe is None:
        raise OMRUnavailableError(
            "Reading PDF/image scores requires Audiveris (https://github.com/Audiveris/audiveris) on PATH. "
            "Install it, or upload the score as MusicXML (.musicxml/.mxl) or MIDI instead.")
    path, out_dir = Path(path), Path(out_dir)
    try:
        r = subprocess.run([exe, "-batch", "-export", "-output", str(out_dir), str(path)],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise InputError(f"Reading {path.name} took longer than {timeout}s; try a smaller or cleaner scan.")
    found = sorted(out_dir.rglob("*.mxl"), key=lambda f: f.stat().st_size, reverse=True)
    found = [f for f in found if f.stem.startswith(path.stem)]
    if r.returncode != 0 or not found:
        detail = (r.stderr or r.stdout).strip().splitlines()
        raise InputError(f"Could not recognise any music in {path.name}"
                         + (f" ({detail[-1]})" if detail else "") + ". Is it a clear scan of a score?")
    return found[0]  # utils/convert_file_type.py unzips it
