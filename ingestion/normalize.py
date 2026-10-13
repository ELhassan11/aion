"""Arabic text normalization (P1-T04).

Applies standard Arabic normalizations so that documents and queries
match regardless of orthographic variation. The pipeline normalizes
documents here; queries must be normalized the same way at retrieval
time (P2).

Choices (documented, not arbitrary):
- alef variants (أإآٱ -> ا): safe, always useful.
- tashkeel (diacritics) removal: safe for search; users rarely type them.
- tatweel (ـ) removal: safe, purely decorative.
- ya normalization (ى -> ي): standard in Arabic IR, slight meaning risk.
- ta_marbuta (ة -> ه): LOSSY (e.g. مكتبة vs مكتبه) — opt-in only.
- digits (٠-٩ -> 0-9): safe, unifies numeric matching.
"""

from __future__ import annotations

import re
import unicodedata

_TASHKEEL = re.compile("[\u0617-\u061a\u064b-\u0652\u0656-\u065f\u06d6-\u06ed]")
_TATWEEL = "\u0640"
_ALEF_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
_YA_MAP = str.maketrans({"ى": "ي"})
_TA_MAP = str.maketrans({"ة": "ه"})
_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_arabic(
    text: str,
    *,
    alef: bool = True,
    tashkeel: bool = True,
    tatweel: bool = True,
    ya: bool = True,
    ta_marbuta: bool = False,
    digits: bool = True,
) -> str:
    """Normalize Arabic text. All options default to the safe standard set."""
    text = unicodedata.normalize("NFC", text)
    if tashkeel:
        text = _TASHKEEL.sub("", text)
    if tatweel:
        text = text.replace(_TATWEEL, "")
    if alef:
        text = text.translate(_ALEF_MAP)
    if ya:
        text = text.translate(_YA_MAP)
    if ta_marbuta:
        text = text.translate(_TA_MAP)
    if digits:
        text = text.translate(_DIGIT_MAP)
    return text
