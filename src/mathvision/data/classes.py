"""Canonical class list for math-vision.

Two folds shape the class space, applied in this order:

1. `CROSS_TYPE_MERGES` folds characters whose glyphs are visually
   indistinguishable across letter/digit boundaries. Downstream OCR is
   expected to resolve `I` vs `1` vs `l` from context (line height,
   adjacency, syntactic rules) - trying to classify them from a tight
   character crop is chasing signal that is not in the pixels.
2. `CASE_MERGED` folds uppercase and lowercase forms that share a glyph.
   Aligned with the EMNIST balanced merge set.

The class list is *derived* from `canonicalize()` over every input
character we care about - so there is exactly one place to change if
the fold rules ever move.
"""

from __future__ import annotations

CROSS_TYPE_MERGES: dict[str, str] = {
    # Vertical-stick class. Canonical: "I".
    "l": "I",
    "L": "I",
    "1": "I",
    # Round class. Canonical: "O" (o folds to O via case merge below).
    "0": "O",
}

CASE_MERGED = frozenset("CIJKMOPSUVWXYZ")

DIGITS = "0123456789"
LETTERS_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LETTERS_LOWER = "abcdefghijklmnopqrstuvwxyz"
PUNCTUATION = "!#$%&()*+,-./:;<=>?@[]^_{|}~"


def canonicalize(ch: str) -> str:
    """Fold a raw character to its canonical class glyph."""
    if len(ch) != 1:
        raise ValueError(f"expected single character, got {ch!r}")
    if ch in CROSS_TYPE_MERGES:
        return CROSS_TYPE_MERGES[ch]
    if ch.isalpha() and ch.upper() in CASE_MERGED:
        return ch.upper()
    return ch


def _build_classes() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for c in DIGITS + LETTERS_UPPER + LETTERS_LOWER + PUNCTUATION:
        canonical = canonicalize(c)
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
    return out


CLASSES: list[str] = _build_classes()
NUM_CLASSES: int = len(CLASSES)
char_to_idx: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}


def idx_to_char(i: int) -> str:
    return CLASSES[i]
