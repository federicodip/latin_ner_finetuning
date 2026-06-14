"""TDD: subword<->word label alignment via fast-tokenizer word_ids()."""

import pytest

from latin_ner.align import align_labels_to_subwords, decode_predictions


class TestAlignLabelsToSubwords:
    def test_first_subword_only_default(self) -> None:
        # word_ids: [CLS, w0, w1, w1, SEP]; w1 has 2 subwords.
        word_ids = [None, 0, 1, 1, None]
        word_label_ids = [3, 5]  # B-GEO, B-GRP
        out = align_labels_to_subwords(word_ids, word_label_ids)
        assert out == [-100, 3, 5, -100, -100]

    def test_label_all_subwords_repeats_label(self) -> None:
        word_ids = [None, 0, 1, 1, None]
        word_label_ids = [3, 5]
        out = align_labels_to_subwords(word_ids, word_label_ids, label_all_subwords=True)
        assert out == [-100, 3, 5, 5, -100]

    def test_custom_ignore_index(self) -> None:
        out = align_labels_to_subwords([None, 0], [4], ignore_index=-1)
        assert out == [-1, 4]

    def test_empty(self) -> None:
        assert align_labels_to_subwords([], []) == []

    def test_all_special_tokens(self) -> None:
        assert align_labels_to_subwords([None, None], []) == [-100, -100]

    def test_word_id_beyond_labels_raises(self) -> None:
        with pytest.raises(IndexError):
            align_labels_to_subwords([0, 1], [7])  # word id 1 but only 1 label


class TestDecodePredictions:
    def test_takes_first_subword_prediction(self) -> None:
        word_ids = [None, 0, 1, 1, None]
        preds = [0, 3, 5, 2, 0]
        assert decode_predictions(word_ids, preds) == [3, 5]

    def test_skips_specials_and_continuations(self) -> None:
        word_ids = [None, 0, 0, 1, None]
        preds = [9, 1, 2, 4, 9]
        assert decode_predictions(word_ids, preds) == [1, 4]

    def test_empty(self) -> None:
        assert decode_predictions([], []) == []
