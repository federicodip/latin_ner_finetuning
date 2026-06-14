"""Parse Herodotos ``.crf`` and generic CoNLL/BIO files into sentences.

Herodotos gold ``.crf`` quirks (verified against the live repo):
  * columns are **label-first**: ``LABEL<TAB>TOKEN``;
  * outside label is the digit ``0`` (not the letter ``O``);
  * entity tags use **suffix** notation ``PRS-B`` / ``PRS-I`` (not ``B-PRS``);
  * a blank line separates sentences.

:func:`normalize_label` maps these to canonical IOB2 (``B-PRS``, ``I-PRS``,
``O``). The scheme is IOB2 (every entity starts with ``-B``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sentence:
    """A tokenized sentence with one canonical IOB2 label per token."""

    tokens: list[str]
    labels: list[str]


def normalize_label(raw: str) -> str:
    """Normalize a raw tag to canonical IOB2.

    Accepts ``0``/``O`` (-> ``O``), suffix form ``PRS-B`` (-> ``B-PRS``), and
    already-canonical ``B-PRS``. Raises ``ValueError`` on anything else.
    """
    s = raw.strip()
    if s in {"0", "O"}:
        return "O"
    if s[:2] in {"B-", "I-"}:
        return s
    if s[-2:] in {"-B", "-I"}:
        etype, bio = s[:-2], s[-1]
        return f"{bio}-{etype}"
    raise ValueError(f"Unrecognized label: {raw!r}")


def _parse_columns(text: str, token_col: int, label_col: int) -> list[Sentence]:
    sentences: list[Sentence] = []
    tokens: list[str] = []
    labels: list[str] = []

    def flush() -> None:
        if tokens:
            sentences.append(Sentence(tokens=tokens.copy(), labels=labels.copy()))
            tokens.clear()
            labels.clear()

    for raw_line in text.split("\n"):
        line = raw_line.rstrip("\r")
        if not line.strip():
            flush()
            continue
        parts = line.split("\t")
        tokens.append(parts[token_col])
        labels.append(normalize_label(parts[label_col]))
    flush()
    return sentences


def parse_crf(text: str) -> list[Sentence]:
    """Parse Herodotos ``.crf`` text (label-first, suffix-BIO)."""
    return _parse_columns(text, token_col=1, label_col=0)


def parse_conll(text: str, token_col: int = 0, label_col: int = 1) -> list[Sentence]:
    """Parse token-first CoNLL/BIO text. ``token_col``/``label_col`` select
    columns for multi-column formats (e.g. 5-col LASLA TSV)."""
    return _parse_columns(text, token_col=token_col, label_col=label_col)


def repair_iob2(labels: list[str]) -> list[str]:
    """Promote any ``I-X`` that does not continue a same-type span to ``B-X``.

    Guards against the handful of stray annotation errors in the gold data so
    seqeval strict scoring sees well-formed IOB2.
    """
    out: list[str] = []
    prev_type: str | None = None
    for lab in labels:
        if lab == "O":
            out.append("O")
            prev_type = None
            continue
        bio, etype = lab.split("-", 1)
        if bio == "I" and prev_type != etype:
            out.append(f"B-{etype}")
        else:
            out.append(lab)
        prev_type = etype
    return out
