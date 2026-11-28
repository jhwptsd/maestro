"""Draw the app icon (src/assets/icon.png, 1024 x 1024): the accent gradient with a sound-wave of bars, as in the app's header.

    python src/scripts/make_icon.py        # needs Pillow

`flet build` turns src/assets/icon.png into the icon of every platform. Rerun this only to change the design.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SIZE = 1024
ACCENT, ACCENT_2 = (124, 92, 255), (56, 182, 255)         # theme.ACCENT, theme.ACCENT_2
BARS = (0.30, 0.52, 0.80, 0.55, 1.00, 0.62, 0.74, 0.46, 0.26)


def draw():
    t = (np.add.outer(np.arange(SIZE), np.arange(SIZE)) / (2 * SIZE - 2))[..., None]          # diagonal 0..1
    gradient = (np.array(ACCENT) * (1 - t) + np.array(ACCENT_2) * t).astype(np.uint8)
    image = Image.fromarray(gradient, "RGB").convert("RGBA")
    mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, SIZE - 1, SIZE - 1), radius=SIZE // 4.5, fill=255)       # the rounded square iOS and macOS expect
    image.putalpha(mask)
    d, width, gap = ImageDraw.Draw(image), 62, 36
    left = (SIZE - (len(BARS) * width + (len(BARS) - 1) * gap)) // 2
    for k, h in enumerate(BARS):
        x, half = left + k * (width + gap), h * SIZE * 0.30
        d.rounded_rectangle((x, SIZE / 2 - half, x + width, SIZE / 2 + half), radius=width // 2, fill=(255, 255, 255, 255))
    return image


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "assets" / "icon.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    draw().save(out)
    print("wrote", out)
