"""TDD: entity-level strict (seqeval) + relaxed scoring, report, and I/O."""

from __future__ import annotations

import json
from pathlib import Path

from latin_ner.evaluate import (
    SANITY_MACRO_F1,
    build_report,
    evaluate_split,
    load_jsonl,
    relaxed_scores,
    render_markdown,
    strict_scores,
    write_eval_outputs,
)


class TestStrictScores:
    def test_perfect(self) -> None:
        g = [["B-PRS", "I-PRS", "O"]]
        out = strict_scores(g, g)
        assert out["macro_f1"] == 1.0
        assert out["per_type"]["PRS"]["f1"] == 1.0
        assert out["per_type"]["GEO"]["support"] == 0

    def test_broken_span_scores_zero_not_one(self) -> None:
        # The whole reason for seqeval strict IOB2: a boundary miss is NOT 1.0.
        g = [["B-PRS", "I-PRS"]]
        p = [["B-PRS", "O"]]
        out = strict_scores(g, p)
        assert out["per_type"]["PRS"]["f1"] == 0.0

    def test_values_are_json_native_floats(self) -> None:
        out = strict_scores([["B-GEO"]], [["B-GEO"]])
        # must be plain float/int (not numpy) so json.dumps works downstream
        assert type(out["macro_f1"]) is float
        assert type(out["per_type"]["GEO"]["support"]) is int


class TestRelaxedScores:
    def test_relaxed_is_at_least_strict_on_boundary_miss(self) -> None:
        g = [["B-PRS", "I-PRS"]]
        p = [["B-PRS", "O"]]  # overlap -> relaxed credits it, strict does not
        strict = strict_scores(g, p)
        relaxed = relaxed_scores(g, p)
        assert relaxed["per_type"]["PRS"]["f1"] == 1.0
        assert strict["per_type"]["PRS"]["f1"] == 0.0


class TestEvaluateSplit:
    def test_counts_entities(self) -> None:
        g = [["B-PRS", "I-PRS", "O", "B-GEO"], ["B-GRP"]]
        p = [["B-PRS", "I-PRS", "O", "O"], ["B-GRP"]]
        s = evaluate_split("in_domain_test", g, p)
        assert s["name"] == "in_domain_test"
        assert s["n_sentences"] == 2
        assert s["n_gold_entities"] == 3  # PRS, GEO, GRP
        assert s["n_pred_entities"] == 2
        assert "strict" in s and "relaxed" in s


class TestBuildAndRender:
    def _split(self) -> dict:
        g = [["B-PRS", "I-PRS", "O", "B-GEO", "B-GRP"]]
        return evaluate_split("in_domain_test", g, g)

    def test_build_report_structure_and_acceptance(self) -> None:
        repro = {"backbone": "latincy/latin-bert", "git_sha": "abc123"}
        rep = build_report([self._split()], repro=repro)
        assert rep["repro"]["backbone"] == "latincy/latin-bert"
        assert "in_domain_test" in rep["splits"]
        # perfect scores -> macro_f1 1.0 -> passes the sanity gate
        assert rep["acceptance"]["in_domain_macro_f1"] == 1.0
        assert rep["acceptance"]["passes"] is True
        assert rep["acceptance"]["threshold"] == SANITY_MACRO_F1

    def test_render_markdown_is_grepable(self) -> None:
        md = render_markdown(build_report([self._split()], repro={"git_sha": "x"}))
        assert "in_domain_test" in md
        assert "STRICT" in md
        assert "macro-F1" in md
        assert "PRS" in md and "GEO" in md and "GRP" in md


class TestIO:
    def test_write_eval_outputs_roundtrips(self, tmp_path: Path) -> None:
        g = [["B-PRS", "O"]]
        rep = build_report([evaluate_split("in_domain_test", g, g)], repro={"git_sha": "x"})
        jp = tmp_path / "eval.json"
        mp = tmp_path / "eval.md"
        write_eval_outputs(rep, jp, mp)
        assert json.loads(jp.read_text(encoding="utf-8"))["acceptance"]["passes"] is True
        assert "STRICT" in mp.read_text(encoding="utf-8")

    def test_load_jsonl(self, tmp_path: Path) -> None:
        p = tmp_path / "test.jsonl"
        p.write_text(
            '{"tokens": ["a", "b"], "labels": ["O", "B-PRS"]}\n'
            '{"tokens": ["c"], "labels": ["B-GEO"]}\n',
            encoding="utf-8",
        )
        tokens, labels = load_jsonl(p)
        assert tokens == [["a", "b"], ["c"]]
        assert labels == [["O", "B-PRS"], ["B-GEO"]]
