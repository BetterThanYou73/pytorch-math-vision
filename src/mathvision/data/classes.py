"""Canonical class list for math-vision.

Uppercase and lowercase letters whose glyphs are visually identical in
tight character crops are merged into a single class, using the uppercase
glyph as canonical. This removes intrinsic label noise that no model can
resolve from a single-character crop.
"""

from __future__ import annotations

CASE_MERGED = frozenset("CIJKLMOPSUVWXYZ")

DIGITS = "0123456789"
LETTERS_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LETTERS_LOWER = "abcdefghijklmnopqrstuvwxyz"
PUNCTUATION = "!#$%&()*+,-./:;<=>?@[]^_{|}~"


def _build_classes() -> list[str]:
    out: list[str] = []
    for c in DIGITS:
        out.append(c)
    for c in LETTERS_UPPER:
        out.append(c)
    for c in LETTERS_LOWER:
        if c.upper() in CASE_MERGED:
            continue
        out.append(c)
    for c in PUNCTUATION:
        out.append(c)
    return out


CLASSES: list[str] = _build_classes()
NUM_CLASSES: int = len(CLASSES)
char_to_idx: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}


def canonicalize(ch: str) -> str:
    """Fold a raw character to its canonical class glyph."""
    if len(ch) != 1:
        raise ValueError(f"expected single character, got {ch!r}")
    if ch.isalpha() and ch.upper() in CASE_MERGED:
        return ch.upper()
    return ch


def idx_to_char(i: int) -> str:
    return CLASSES[i]
