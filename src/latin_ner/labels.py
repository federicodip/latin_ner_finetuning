"""BIO label set for classical-Latin NER and label <-> id encoding.

The label order is part of the OUTPUT CONTRACT: it defines ``id2label`` /
``label2id`` baked into the checkpoint's ``config.json``. Do not reorder.
"""

from __future__ import annotations

from collections.abc import Iterable

#: Native Herodotos entity types, in contract order (person / place / group).
ENTITY_TYPES: tuple[str, str, str] = ("PRS", "GEO", "GRP")

#: The 7 BIO labels, in contract order. id == index in this list.
LABELS: list[str] = [
    "O",
    "B-PRS",
    "I-PRS",
    "B-GEO",
    "I-GEO",
    "B-GRP",
    "I-GRP",
]

NUM_LABELS: int = len(LABELS)

LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = dict(enumerate(LABELS))


def label_to_id(label: str) -> int:
    """Return the id for ``label``; raise ``KeyError`` if unknown."""
    return LABEL2ID[label]


def id_to_label(idx: int) -> str:
    """Return the label for ``idx``; raise ``KeyError`` if out of range."""
    return ID2LABEL[idx]


def encode(labels: Iterable[str]) -> list[int]:
    """Encode a sequence of BIO labels to ids."""
    return [LABEL2ID[label] for label in labels]


def decode(ids: Iterable[int]) -> list[str]:
    """Decode a sequence of ids back to BIO labels."""
    return [ID2LABEL[i] for i in ids]
