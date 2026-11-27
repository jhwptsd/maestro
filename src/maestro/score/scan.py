# Clean photographed pages of sheet music for the music reader (new file).
#
# The app scans a score with the phone's camera: one photo per page. A photo has shadows and uneven light, which Audiveris (score/omr.py) reads badly,
# so each page is shrunk to at most 2400 px on its long side, turned to grey and thresholded locally (Bradley's adaptive method over an integral image:
# a pixel is ink when it is 12 % darker than the mean of its surroundings), which leaves black music on white paper whatever the lighting. One page becomes a
# JPEG, several a multi-page PDF (Pillow writes it), which utils/convert_file_type.py and Audiveris accept like any uploaded score.
# Needs Pillow (pip install pillow); numpy does the arithmetic.
import io
from pathlib import Path

import numpy as np

from maestro.utils.errors import InputError, ToolMissingError

LONG_SIDE = 2400
INK = 0.12                       # a pixel this much darker than its surroundings is ink


def _pillow():
    try:
        from PIL import Image
    except ImportError:
        raise ToolMissingError("Scanning a score needs Pillow: run  pip install pillow.") from None
    return Image


def clean_page(photo):
    """The bytes (or path) of a photo of a page -> a PIL 'L' image of black ink on white paper."""
    Image = _pillow()
    try:
        image = Image.open(io.BytesIO(photo) if isinstance(photo, (bytes, bytearray)) else photo).convert("L")
    except OSError:
        raise InputError("That photo could not be read as an image.") from None
    scale = min(1.0, LONG_SIDE / max(image.size))
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    grey = np.asarray(image, dtype=np.float64)
    h, w = grey.shape
    integral = np.zeros((h + 1, w + 1))
    integral[1:, 1:] = grey.cumsum(0).cumsum(1)
    r = max(8, max(w, h) // 32)                                  # window: about a thirty-second of the page on each side
    y0, y1 = np.clip(np.arange(h) - r, 0, h), np.clip(np.arange(h) + r + 1, 0, h)
    x0, x1 = np.clip(np.arange(w) - r, 0, w), np.clip(np.arange(w) + r + 1, 0, w)
    window = integral[np.ix_(y1, x1)] - integral[np.ix_(y0, x1)] - integral[np.ix_(y1, x0)] + integral[np.ix_(y0, x0)]
    mean = window / ((y1 - y0)[:, None] * (x1 - x0)[None, :])
    return Image.fromarray(np.where(grey < mean * (1 - INK), 0, 255).astype(np.uint8), "L")


def pages_to_score_file(photos, out_dir, name="score"):
    """Photos of the pages (bytes or paths) -> the path of one file for the analysis: `name`.jpg for one page, `name`.pdf for several."""
    if not photos:
        raise InputError("Take a photo of at least one page first.")
    pages = [clean_page(p) for p in photos]
    out = Path(out_dir) / (name + (".jpg" if len(pages) == 1 else ".pdf"))
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(pages) == 1:
        pages[0].save(out, "JPEG", quality=92)
    else:
        pages[0].save(out, "PDF", save_all=True, append_images=pages[1:], resolution=200.0)
    return out
