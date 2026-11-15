# Optical Music Recognition (OMR) module for processing and analyzing sheet music images.
#
# Only used when the score is an image/PDF: converts it to MusicXML, which then continues
# through preprocessing/score.py -> score/musicxml.py.
# Use Audiveris and Partitura
#
# OMR is delegated to the external Audiveris program (https://github.com/Audiveris/audiveris),
# which must be installed and on PATH as `audiveris`. Quality depends on scan quality; always
# eyeball the result (see score/visualize.py).
import shutil
import subprocess
from pathlib import Path


def run_omr(path, out_dir):
    """Run Audiveris on an image/PDF -> path of the exported MusicXML (.mxl) file."""
    exe = shutil.which("audiveris")
    if exe is None:
        raise RuntimeError("OMR requires Audiveris on PATH; alternatively provide a MusicXML score.")
    subprocess.run([exe, "-batch", "-export", "-output", str(out_dir), str(path)], check=True)
    found = sorted(Path(out_dir).rglob(f"{Path(path).stem}*.mxl"))
    if not found:
        raise RuntimeError("Audiveris produced no MusicXML output.")
    return found[0]  # utils/convert_file_type.py unzips it
