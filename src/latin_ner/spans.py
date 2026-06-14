"""BIO -> typed-span decoding and entity-level strict/relaxed P/R/F1.

The *official* strict number in :mod:`latin_ner.evaluate` is computed with
seqeval (``mode="strict", scheme="IOB2"``). The pure scorers here power the
**relaxed** metric and unit tests, and cross-check seqeval's strict number.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .labels import ENTITY_TYPES


@dataclass(frozen=True)
class Span:
    """A typed entity span over token indices, ``[start, end)``."""

    start: int
    end: int
    etype: str


@dataclass(frozen=True)
class PRF:
    """Precision/recall/F1 plus the counts they were derived from."""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    support: int


@dataclass(frozen=True)
class ScoreReport:
    """Per-type PRF, micro-averaged PRF, and macro-averaged F1."""

    per_type: dict[str, PRF]
    micro: PRF
    macro_f1: float


def bio_to_spans(labels: Sequence[str]) -> list[Span]:
    """Decode a BIO sequence into typed spans (tolerant: a stray ``I-`` opens
    a span). Used to extract predicted spans; strict scoring is seqeval's job.
    """
    spans: list[Span] = []
    start = 0
    etype: str | None = None
    for i, lab in enumerate(labels):
        if lab == "O":
            if etype is not None:
                spans.append(Span(start, i, etype))
                etype = None
            continue
        bio, t = lab.split("-", 1)
        if bio == "B" or t != etype:
            if etype is not None:
                spans.append(Span(start, i, etype))
            start, etype = i, t
    if etype is not None:
        spans.append(Span(start, len(labels), etype))
    return spans


def prf_from_counts(tp: int, fp: int, fn: int) -> PRF:
    """Build a :class:`PRF` from raw counts (0 on any zero division)."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF(precision, recall, f1, tp, fp, fn, tp + fn)


def _report(
    tp: dict[str, int],
    fp: dict[str, int],
    fn: dict[str, int],
    types: Sequence[str],
) -> ScoreReport:
    per_type = {t: prf_from_counts(tp[t], fp[t], fn[t]) for t in types}
    micro = prf_from_counts(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    macro_f1 = sum(per_type[t].f1 for t in types) / len(types)
    return ScoreReport(per_type=per_type, micro=micro, macro_f1=macro_f1)


def score_strict_spans(
    gold_seqs: Sequence[Sequence[str]],
    pred_seqs: Sequence[Sequence[str]],
    types: Sequence[str] = ENTITY_TYPES,
) -> ScoreReport:
    """Strict entity-level scoring: a span is correct iff (start, end, type)
    match exactly."""
    if len(gold_seqs) != len(pred_seqs):
        raise ValueError("gold_seqs and pred_seqs must have the same length")
    tp = {t: 0 for t in types}
    fp = {t: 0 for t in types}
    fn = {t: 0 for t in types}
    for gold, pred in zip(gold_seqs, pred_seqs, strict=True):
        gset = {(s.start, s.end, s.etype) for s in bio_to_spans(gold)}
        pset = {(s.start, s.end, s.etype) for s in bio_to_spans(pred)}
        for t in types:
            gt = {x for x in gset if x[2] == t}
            pt = {x for x in pset if x[2] == t}
            tp[t] += len(gt & pt)
            fp[t] += len(pt - gt)
            fn[t] += len(gt - pt)
    return _report(tp, fp, fn, types)


def _overlaps(a: Span, b: Span) -> bool:
    return a.start < b.end and b.start < a.end


def score_relaxed(
    gold_seqs: Sequence[Sequence[str]],
    pred_seqs: Sequence[Sequence[str]],
    types: Sequence[str] = ENTITY_TYPES,
) -> ScoreReport:
    """Relaxed scoring: a predicted span matches a gold span iff same type and
    any token overlap. Matching is one-to-one (greedy)."""
    if len(gold_seqs) != len(pred_seqs):
        raise ValueError("gold_seqs and pred_seqs must have the same length")
    tp = {t: 0 for t in types}
    fp = {t: 0 for t in types}
    fn = {t: 0 for t in types}
    for gold, pred in zip(gold_seqs, pred_seqs, strict=True):
        gspans = bio_to_spans(gold)
        pspans = bio_to_spans(pred)
        matched: set[int] = set()
        for ps in pspans:
            hit: int | None = None
            for gi, gs in enumerate(gspans):
                if gi not in matched and gs.etype == ps.etype and _overlaps(ps, gs):
                    hit = gi
                    break
            if hit is None:
                fp[ps.etype] += 1
            else:
                matched.add(hit)
                tp[ps.etype] += 1
        for gi, gs in enumerate(gspans):
            if gi not in matched:
                fn[gs.etype] += 1
    return _report(tp, fp, fn, types)
