"""Subword <-> word label alignment via a fast tokenizer's ``word_ids()``.

The Herodotos data is already word-tokenized, so we tokenize with
``is_split_into_words=True`` and use the fast LatinBERT tokenizer's
``word_ids()`` to (a) spread word labels onto the first subword of each word
for training, and (b) read predictions back off the first subword per word.
"""

from __future__ import annotations

from collections.abc import Sequence


def align_labels_to_subwords(
    word_ids: Sequence[int | None],
    word_label_ids: Sequence[int],
    *,
    label_all_subwords: bool = False,
    ignore_index: int = -100,
) -> list[int]:
    """Map word-level label ids onto subword tokens.

    Special tokens (``word_id is None``) and, by default, the non-first
    subwords of a word receive ``ignore_index`` so they are excluded from the
    loss. With ``label_all_subwords=True`` every subword of a word repeats the
    word's label id.
    """
    out: list[int] = []
    prev: int | None = None
    for wid in word_ids:
        if wid is None:
            out.append(ignore_index)
            prev = None
        elif wid != prev:
            out.append(word_label_ids[wid])
            prev = wid
        else:
            out.append(word_label_ids[wid] if label_all_subwords else ignore_index)
    return out


def decode_predictions(
    word_ids: Sequence[int | None],
    pred_label_ids: Sequence[int],
) -> list[int]:
    """Collapse per-subword predictions to one prediction per word by taking
    the first subword of each word (special tokens skipped)."""
    out: list[int] = []
    prev: int | None = None
    for wid, pred in zip(word_ids, pred_label_ids, strict=True):
        if wid is None:
            prev = None
            continue
        if wid != prev:
            out.append(pred)
            prev = wid
    return out
