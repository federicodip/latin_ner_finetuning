"""Character-offset emission for NER on raw text.

The OUTPUT CONTRACT requires native PRS/GEO/GRP entities with **character
offsets**. Rather than depend on the custom LatinBERT tokenizer's
``offset_mapping`` (whose escaping pre-tokenizer makes it brittle), we track
each whitespace word's char span ourselves, predict per word via
``word_ids()``, then convert token-index spans back to char offsets. This is
exact and tokenizer-independent.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_WORD_RE = re.compile(r"\S+")


def whitespace_word_spans(text: str) -> list[tuple[str, int, int]]:
    """Split ``text`` on whitespace into ``(word, start, end)`` char spans."""
    return [(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def spans_to_char_offsets(
    text: str,
    word_spans: Sequence[tuple[str, int, int]],
    token_spans: Sequence[tuple[int, int, str]],
) -> list[dict[str, object]]:
    """Map token-index entity spans ``[start, end)`` to character offsets.

    ``word_spans`` is the output of :func:`whitespace_word_spans`; each
    ``token_spans`` entry is ``(start_word, end_word, etype)``.
    """
    out: list[dict[str, object]] = []
    for start_word, end_word, etype in token_spans:
        char_start = word_spans[start_word][1]
        char_end = word_spans[end_word - 1][2]
        out.append(
            {
                "text": text[char_start:char_end],
                "start": char_start,
                "end": char_end,
                "type": etype,
            }
        )
    return out
