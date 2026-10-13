"""Generate the OCR accuracy test image (P1-T03).

Renders known Arabic text to a clean PNG + writes the expected text file.
Requires an Arabic font (Noto Sans Arabic used here) and libraqm for shaping.

Usage:
    python scripts/make_ocr_test_image.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT_IMAGE = ROOT / "data" / "sample" / "ocr_test.png"
OUT_TEXT = ROOT / "data" / "sample" / "ocr_test_expected.txt"

LINES = [
    "اختبار دقة التعرف الضوئي على الحروف",
    "متجر النيل يقدم أفضل المنتجات بأسعار منافسة",
    "هاتف ذكي بسعر 8500 جنيه والكمية المتاحة 12",
    "للتواصل: 01234567890",
]

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansArabic-CondensedBlack.ttf",  # Linux (Noto)
    "C:/Windows/Fonts/arial.ttf",  # Windows (Arial has Arabic glyphs)
    "C:/Windows/Fonts/trado.ttf",  # Windows (Traditional Arabic)
]


def _find_font(explicit: str | None) -> str:
    if explicit:
        return explicit
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise RuntimeError(
        "No Arabic font found. Pass one explicitly: "
        "python scripts/make_ocr_test_image.py --font /path/to/arabic-font.ttf"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the OCR accuracy test image")
    ap.add_argument("--font", default=None, help="path to an Arabic TTF font")
    args = ap.parse_args()

    font = ImageFont.truetype(_find_font(args.font), 56)
    # Measure to size the canvas.
    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)
    widths = [draw.textlength(line, font=font) for line in LINES]
    width = int(max(widths)) + 120
    line_height = 95
    height = line_height * len(LINES) + 80

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = 40
    for line in LINES:
        draw.text((60, y), line, font=font, fill="black", direction="rtl", language="ar")
        y += line_height

    OUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT_IMAGE)
    OUT_TEXT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"wrote {OUT_IMAGE} ({width}x{height})")
    print(f"wrote {OUT_TEXT}")


if __name__ == "__main__":
    main()
