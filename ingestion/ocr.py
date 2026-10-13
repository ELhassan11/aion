"""Arabic OCR (P1-T03) via Tesseract.

Requires the Tesseract binary with Arabic language data, plus
pytesseract and pillow (ingestion/requirements.txt).
"""

from __future__ import annotations

import shutil
from pathlib import Path

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None


def _check_deps() -> None:
    if pytesseract is None or Image is None:
        raise RuntimeError(
            "missing Python deps: pip install -r ingestion/requirements.txt"
        )
    if shutil.which("tesseract") is None:
        raise RuntimeError(
            "Tesseract binary not found. Windows: install from "
            "https://github.com/UB-Mannheim/tesseract/wiki "
            "and include the Arabic language data during setup."
        )


def ocr_image(path: Path, lang: str = "ara") -> str:
    """Run OCR on an image file and return the extracted text."""
    _check_deps()
    return pytesseract.image_to_string(Image.open(path), lang=lang)
