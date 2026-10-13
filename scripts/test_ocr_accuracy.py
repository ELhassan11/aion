"""OCR accuracy test (P1-T03).

Runs Tesseract on the generated test image and compares the output
against the expected text using character-level accuracy
(1 - levenshtein_distance / len(expected)).

Usage:
    python scripts/test_ocr_accuracy.py

Prints ASCII-only output so it renders fine in any terminal.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingestion.ocr import ocr_image

TEST_IMAGE = ROOT / "data" / "sample" / "ocr_test.png"
EXPECTED_FILE = ROOT / "data" / "sample" / "ocr_test_expected.txt"

# >= 0.80 means OCR is usable for our ingestion pipeline.
THRESHOLD = 0.80


def _normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_accuracy(expected: str, actual: str) -> float:
    expected, actual = _normalize(expected), _normalize(actual)
    if not expected:
        return 0.0
    return max(0.0, 1.0 - levenshtein(expected, actual) / len(expected))


def main() -> None:
    if not TEST_IMAGE.exists():
        raise RuntimeError(
            f"test image not found: {TEST_IMAGE}\n"
            "generate it first: python scripts/make_ocr_test_image.py"
        )
    expected = EXPECTED_FILE.read_text(encoding="utf-8")
    actual = ocr_image(TEST_IMAGE)
    acc = char_accuracy(expected, actual)
    print(f"expected chars: {len(_normalize(expected))}")
    print(f"ocr chars:      {len(_normalize(actual))}")
    print(f"accuracy:       {acc:.2f}")
    print(f"verdict:        {'PASS - OCR usable' if acc >= THRESHOLD else 'FAIL - OCR not reliable enough'}")


if __name__ == "__main__":
    main()
