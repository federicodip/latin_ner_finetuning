"""TDD: BIO->span decoding and strict/relaxed entity-level P/R/F1."""

import pytest

from latin_ner.spans import (
    PRF,
    Span,
    bio_to_spans,
    prf_from_counts,
    score_relaxed,
    score_strict_spans,
)


class TestBioToSpans:
    def test_single_token_span(self) -> None:
        assert bio_to_spans(["B-PRS"]) == [Span(0, 1, "PRS")]

    def test_multitoken_span(self) -> None:
        assert bio_to_spans(["B-PRS", "I-PRS", "O"]) == [Span(0, 2, "PRS")]

    def test_adjacent_same_type_are_two_spans(self) -> None:
        assert bio_to_spans(["B-PRS", "B-PRS"]) == [Span(0, 1, "PRS"), Span(1, 2, "PRS")]

    def test_two_types(self) -> None:
        assert bio_to_spans(["B-GEO", "B-GRP", "I-GRP"]) == [
            Span(0, 1, "GEO"),
            Span(1, 3, "GRP"),
        ]

    def test_all_outside_is_empty(self) -> None:
        assert bio_to_spans(["O", "O"]) == []

    def test_lenient_leading_i_opens_span(self) -> None:
        # Tolerant decoder (used to extract predicted spans): a stray I- opens
        # a span. Official strict scoring is delegated to seqeval separately.
        assert bio_to_spans(["I-PRS", "I-PRS"]) == [Span(0, 2, "PRS")]


class TestPrfFromCounts:
    def test_basic(self) -> None:
        p = prf_from_counts(tp=8, fp=2, fn=2)
        assert isinstance(p, PRF)
        assert p.precision == pytest.approx(0.8)
        assert p.recall == pytest.approx(0.8)
        assert p.f1 == pytest.approx(0.8)
        assert p.support == 10  # tp + fn

    def test_zero_division_is_zero(self) -> None:
        p = prf_from_counts(tp=0, fp=0, fn=0)
        assert (p.precision, p.recall, p.f1) == (0.0, 0.0, 0.0)


class TestStrictScoring:
    def test_perfect_match(self) -> None:
        gold = [["B-PRS", "I-PRS", "O"]]
        r = score_strict_spans(gold, gold)
        assert r.micro.f1 == pytest.approx(1.0)
        assert r.per_type["PRS"].f1 == pytest.approx(1.0)
        assert r.macro_f1 == pytest.approx(1.0 / 3)  # GEO,GRP have no support -> f1 0

    def test_boundary_miss_is_wrong(self) -> None:
        gold = [["B-PRS", "I-PRS"]]
        pred = [["B-PRS", "O"]]  # predicted (0,1) != gold (0,2)
        r = score_strict_spans(gold, pred)
        assert r.per_type["PRS"].tp == 0
        assert r.per_type["PRS"].fp == 1
        assert r.per_type["PRS"].fn == 1

    def test_macro_f1_averages_present_types(self) -> None:
        gold = [["B-PRS", "B-GEO", "B-GRP"]]
        pred = [["B-PRS", "O", "O"]]  # PRS f1=1, GEO f1=0, GRP f1=0
        r = score_strict_spans(gold, pred)
        assert r.macro_f1 == pytest.approx(1.0 / 3)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            score_strict_spans([["O"]], [["O"], ["O"]])


class TestRelaxedScoring:
    def test_overlap_counts_as_match(self) -> None:
        gold = [["B-PRS", "I-PRS"]]
        pred = [["B-PRS", "O"]]  # overlaps gold span, same type -> relaxed TP
        r = score_relaxed(gold, pred)
        assert r.per_type["PRS"].tp == 1
        assert r.per_type["PRS"].fp == 0
        assert r.per_type["PRS"].fn == 0

    def test_type_mismatch_is_not_a_match(self) -> None:
        gold = [["B-PRS"]]
        pred = [["B-GEO"]]
        r = score_relaxed(gold, pred)
        assert r.per_type["PRS"].fn == 1
        assert r.per_type["GEO"].fp == 1

    def test_one_to_one_matching(self) -> None:
        # one gold span, two overlapping predicted spans -> 1 TP, 1 FP
        gold = [["B-PRS", "I-PRS", "I-PRS"]]
        pred = [["B-PRS", "O", "B-PRS"]]
        r = score_relaxed(gold, pred)
        assert r.per_type["PRS"].tp == 1
        assert r.per_type["PRS"].fp == 1

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            score_relaxed([["O"]], [["O"], ["O"]])
