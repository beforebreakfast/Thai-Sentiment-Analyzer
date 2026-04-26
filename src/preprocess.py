"""Lightweight preprocessing for Thai/English sentiment text.

Important assignment rule: do NOT over-clean. We preserve emoji, slang,
negation words, and punctuation that can carry sentiment.
"""
from __future__ import annotations

import re
import unicodedata

_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_MULTI_SPACE_RE = re.compile(r"\s+")
_LATIN_RE = re.compile(r"[A-Za-z]")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # emoji/symbols
    "\U00002700-\U000027BF"
    "]+",
    flags=re.UNICODE,
)

SLANG_TERMS = [
    "เรย", "งง", "555", "5555", "โคตร", "เว่อ", "จึ้ง", "ปัง", "ดีงาม",
    "ไม่ไหว", "กาก", "อะ", "อ่ะ", "น้า", "นะคะ", "แอบ", "เฉยๆ", "ฟิน",
]
CONTRAST_TERMS = ["แต่", "แต่ว่า", "ทว่า", "อย่างไรก็ตาม", "ถึงแม้", "แม้ว่า", "แต่ก็"]
NEGATION_TERMS = ["ไม่", "ไม่ค่อย", "ไม่ได้", "ไม่มี", "ไม่ใช่", "ห่วย", "พัง"]


def normalize_text(text: object) -> str:
    """Apply minimal normalization while preserving useful sentiment signals."""
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    # Lowercase only Latin/English chunks. Thai text is not affected.
    if _LATIN_RE.search(text):
        text = re.sub(r"[A-Za-z]+", lambda m: m.group(0).lower(), text)
    return text


def build_input_text(headline: str = "", body: str = "") -> str:
    headline = normalize_text(headline)
    body = normalize_text(body)
    return normalize_text(f"{headline} {body}".strip())


def detect_noise_and_difficulty(text: object) -> dict[str, object]:
    """Create clean/noisy and easy/hard tags for dataset understanding/error analysis."""
    t = normalize_text(text)
    lower_t = t.lower()
    has_emoji = bool(EMOJI_RE.search(t))
    has_slang = any(term in lower_t for term in SLANG_TERMS)
    has_contrast = any(term in lower_t for term in CONTRAST_TERMS)
    has_negation = any(term in lower_t for term in NEGATION_TERMS)
    is_short = len(t) < 25
    signal_count = sum([has_emoji, has_slang, has_contrast, has_negation, is_short])
    return {
        "clean_noisy": "noisy" if (has_emoji or has_slang) else "clean",
        "easy_hard": "hard" if (has_contrast or has_negation or signal_count >= 2) else "easy",
        "has_emoji": has_emoji,
        "has_slang": has_slang,
        "has_contrast": has_contrast,
        "has_negation": has_negation,
        "is_short": is_short,
    }


def simple_cues(text: object, limit: int = 8) -> list[str]:
    """Return readable cues for the UI without requiring model internals."""
    t = normalize_text(text).lower()
    cues: list[str] = []
    for term in [*SLANG_TERMS, *CONTRAST_TERMS, *NEGATION_TERMS]:
        if term in t and term not in cues:
            cues.append(term)
    if EMOJI_RE.search(t):
        cues.append("emoji")
    return cues[:limit]
