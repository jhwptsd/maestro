# Convert file type for score processing
#
# Used by preprocessing/score.py to normalize input scores (PDF/image -> score/omr.py, or
# MusicXML passthrough) before score/musicxml.py parses them.
#
# Supported inputs: .musicxml/.xml (passthrough), .mxl (compressed MusicXML, unzipped here),
# .pdf/.png/.jpg/.jpeg/.tif/.tiff (OMR via score/omr.py).
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

PLAIN_XML = {".musicxml", ".xml"}
IMAGE_LIKE = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def convert_to_musicxml(path, out_dir=None):
    """Return the path of an uncompressed MusicXML file equivalent to the score at `path`."""
    path = Path(path)
    out_dir = Path(out_dir) if out_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()

    if ext in PLAIN_XML:
        return path
    if ext == ".mxl":
        return _unzip_mxl(path, out_dir)
    if ext in IMAGE_LIKE:
        from maestro.score.omr import run_omr  # lazy: only needs Audiveris for this branch
        return convert_to_musicxml(run_omr(path, out_dir), out_dir)  # OMR emits .mxl -> unzip
    raise ValueError(f"Unsupported score format: {ext!r}")


def _unzip_mxl(path, out_dir):
    """.mxl is a zip; META-INF/container.xml names the real MusicXML file inside."""
    with zipfile.ZipFile(path) as z:
        container = ET.fromstring(z.read("META-INF/container.xml"))
        inner = container.find(".//rootfile").attrib["full-path"]
        out = out_dir / (path.stem + ".musicxml")
        out.write_bytes(z.read(inner))
    return out
