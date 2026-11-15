# Convert file type for score processing
#
# Used by preprocessing/score.py to normalize input scores (PDF/image -> score/omr.py, MIDI ->
# utils/midi.py, or MusicXML passthrough) before score/musicxml.py parses them.
#
# The type is detected from the file's *content* (magic bytes), not its extension, because user
# uploads are often misnamed. Supported: MusicXML (.musicxml/.xml), compressed MusicXML (.mxl,
# unzipped here), MIDI (.mid/.midi), PDF and PNG/JPEG/TIFF images (OMR via score/omr.py).
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from maestro.utils.errors import InputError
from maestro.utils.midi import midi_to_musicxml

_MAGIC = [(b"%PDF", "pdf"), (b"\x89PNG", "image"), (b"\xff\xd8\xff", "image"), (b"II*\x00", "image"),
          (b"MM\x00*", "image"), (b"MThd", "midi"), (b"PK\x03\x04", "mxl")]


def detect_score_type(path):
    """-> one of "musicxml", "mxl", "midi", "pdf", "image"; raises InputError if missing/unknown."""
    path = Path(path)
    if not path.is_file():
        raise InputError(f"File not found: {path}")
    head = path.read_bytes()[:512]
    if not head:
        raise InputError(f"{path.name} is empty.")
    for magic, kind in _MAGIC:
        if head.startswith(magic):
            return kind
    if head.lstrip(b"\xef\xbb\xbf \t\r\n").startswith(b"<"):   # (BOM +) markup
        return "musicxml"
    raise InputError(f"Unsupported score format: {path.name!r} (use MusicXML, MIDI, PDF or an image).")


def convert_to_musicxml(path, out_dir=None):
    """Return the path of an uncompressed MusicXML file equivalent to the score at `path`."""
    path = Path(path)
    kind = detect_score_type(path)
    out_dir = Path(out_dir) if out_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if kind == "musicxml":
        return path
    if kind == "mxl":
        return _unzip_mxl(path, out_dir)
    if kind == "midi":
        return midi_to_musicxml(path, out_dir / (path.stem + ".musicxml"))
    from maestro.score.omr import run_omr  # lazy: only needs Audiveris for pdf/image
    return _unzip_mxl(Path(run_omr(path, out_dir)), out_dir)  # OMR emits .mxl


def _unzip_mxl(path, out_dir):
    """.mxl is a zip; META-INF/container.xml names the real MusicXML file inside."""
    try:
        with zipfile.ZipFile(path) as z:
            container = ET.fromstring(z.read("META-INF/container.xml"))
            inner = container.find(".//rootfile").attrib["full-path"]
            out = out_dir / (path.stem + ".musicxml")
            out.write_bytes(z.read(inner))
    except (zipfile.BadZipFile, KeyError, ET.ParseError, AttributeError):
        raise InputError(f"{path.name} is not a valid compressed MusicXML (.mxl) file.")
    return out
