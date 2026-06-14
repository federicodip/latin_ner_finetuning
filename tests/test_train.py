"""TDD: training config + eval-prediction decoding/metrics (no torch needed)."""

from __future__ import annotations

import pytest

from latin_ner.train import (
    TrainConfig,
    build_compute_metrics,
    compute_metrics_from_arrays,
    decode_eval,
)


class TestTrainConfig:
    def test_beersmans_best_in_domain_defaults(self) -> None:
        c = TrainConfig()
        assert c.backbone == "latincy/latin-bert"
        assert c.learning_rate == pytest.approx(7.89e-5)
        assert c.weight_decay == pytest.approx(0.10)
        assert c.num_train_epochs == 3
        assert c.per_device_train_batch_size == 16
        assert c.warmup_ratio == pytest.approx(0.1)
        assert c.seed == 13

    def test_label_all_subwords_off_by_default(self) -> None:
        assert TrainConfig().label_all_subwords is False


class TestDecodeEval:
    def test_drops_ignore_index_positions(self) -> None:
        pred = [[0, 1, 2, 0]]
        labels = [[-100, 1, 2, -100]]
        gold, pred_seqs = decode_eval(pred, labels)
        assert gold == [["B-PRS", "I-PRS"]]
        assert pred_seqs == [["B-PRS", "I-PRS"]]

    def test_maps_ids_to_labels(self) -> None:
        gold, pred_seqs = decode_eval([[3, 5]], [[3, 5]])
        assert gold == [["B-GEO", "B-GRP"]]
        assert pred_seqs == [["B-GEO", "B-GRP"]]

    def test_mismatch_between_pred_and_gold_preserved(self) -> None:
        gold, pred_seqs = decode_eval([[1, 0]], [[1, 2]])
        assert gold == [["B-PRS", "I-PRS"]]
        assert pred_seqs == [["B-PRS", "O"]]


class TestComputeMetrics:
    def test_perfect_macro_f1(self) -> None:
        m = compute_metrics_from_arrays([[1, 2, 0]], [[1, 2, 0]])
        assert m["macro_f1"] == 1.0
        assert set(m) >= {"macro_f1", "micro_f1", "precision", "recall"}

    def test_boundary_miss_is_strict_zero(self) -> None:
        # gold B-PRS I-PRS vs pred B-PRS O -> strict span wrong -> f1 0
        m = compute_metrics_from_arrays([[1, 0]], [[1, 2]])
        assert m["macro_f1"] == 0.0

    def test_ignore_index_excluded(self) -> None:
        m = compute_metrics_from_arrays([[9, 1, 2, 9]], [[-100, 1, 2, -100]])
        assert m["macro_f1"] == 1.0


class TestBuildComputeMetrics:
    def test_returns_callable_consuming_eval_pred_tuple(self) -> None:
        fn = build_compute_metrics()
        out = fn(([[1, 2, 0]], [[1, 2, 0]]))
        assert out["macro_f1"] == 1.0
