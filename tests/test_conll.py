"""TDD: parsing Herodotos .crf (label-first, suffix-BIO) and generic CoNLL."""

import pytest

from latin_ner.conll import (
    Sentence,
    normalize_label,
    parse_conll,
    parse_crf,
    repair_iob2,
)

# Herodotos .crf: LABEL<TAB>TOKEN, outside="0", suffix tags PRS-B/PRS-I,
# blank line = sentence boundary.
CRF_SAMPLE = "GEO-B\tGallia\n0\test\n0\t.\n\nPRS-B\tM.\nPRS-I\tVarrone\n0\t.\n"


class TestNormalizeLabel:
    def test_zero_becomes_o(self) -> None:
        assert normalize_label("0") == "O"

    def test_o_passthrough(self) -> None:
        assert normalize_label("O") == "O"

    def test_suffix_b_flipped(self) -> None:
        assert normalize_label("PRS-B") == "B-PRS"

    def test_suffix_i_flipped(self) -> None:
        assert normalize_label("GEO-I") == "I-GEO"

    def test_canonical_prefix_passthrough(self) -> None:
        assert normalize_label("B-GRP") == "B-GRP"
        assert normalize_label("I-PRS") == "I-PRS"

    def test_surrounding_whitespace_stripped(self) -> None:
        assert normalize_label("  PRS-B \r") == "B-PRS"

    def test_unknown_label_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_label("FOO")


class TestParseCrf:
    def test_produces_two_sentences(self) -> None:
        assert len(parse_crf(CRF_SAMPLE)) == 2

    def test_returns_sentence_objects(self) -> None:
        assert all(isinstance(s, Sentence) for s in parse_crf(CRF_SAMPLE))

    def test_first_sentence_tokens(self) -> None:
        assert parse_crf(CRF_SAMPLE)[0].tokens == ["Gallia", "est", "."]

    def test_first_sentence_labels_normalized(self) -> None:
        assert parse_crf(CRF_SAMPLE)[0].labels == ["B-GEO", "O", "O"]

    def test_multitoken_span_normalized(self) -> None:
        assert parse_crf(CRF_SAMPLE)[1].labels == ["B-PRS", "I-PRS", "O"]

    def test_crlf_line_endings_handled(self) -> None:
        s = parse_crf(CRF_SAMPLE.replace("\n", "\r\n"))
        assert s[0].tokens == ["Gallia", "est", "."]
        assert s[1].labels == ["B-PRS", "I-PRS", "O"]

    def test_trailing_blank_lines_make_no_empty_sentence(self) -> None:
        assert len(parse_crf(CRF_SAMPLE + "\n\n\n")) == 2

    def test_empty_input_is_empty_list(self) -> None:
        assert parse_crf("") == []
        assert parse_crf("\n\n") == []

    def test_no_trailing_newline(self) -> None:
        s = parse_crf("PRS-B\tCicero")
        assert s[0].tokens == ["Cicero"]
        assert s[0].labels == ["B-PRS"]

    def test_tokens_and_labels_align(self) -> None:
        for sent in parse_crf(CRF_SAMPLE):
            assert len(sent.tokens) == len(sent.labels)


class TestParseConll:
    def test_token_first_layout(self) -> None:
        text = "Gallia\tB-GEO\nest\tO\n\nCicero\tB-PRS\n"
        s = parse_conll(text)
        assert s[0].tokens == ["Gallia", "est"]
        assert s[0].labels == ["B-GEO", "O"]
        assert s[1].tokens == ["Cicero"]

    def test_custom_columns(self) -> None:
        # 5-col LASLA-like: id, token, lemma, uri, label
        text = "1\tsemper\tL1\tURI\tO\n2\tCordi\tL2\tURI\tB-PRS\n"
        s = parse_conll(text, token_col=1, label_col=4)
        assert s[0].tokens == ["semper", "Cordi"]
        assert s[0].labels == ["O", "B-PRS"]


class TestRepairIob2:
    def test_stray_leading_i_promoted(self) -> None:
        assert repair_iob2(["I-PRS", "O"]) == ["B-PRS", "O"]

    def test_i_after_different_type_promoted(self) -> None:
        assert repair_iob2(["B-GEO", "I-PRS"]) == ["B-GEO", "B-PRS"]

    def test_valid_continuation_kept(self) -> None:
        assert repair_iob2(["B-PRS", "I-PRS", "I-PRS"]) == ["B-PRS", "I-PRS", "I-PRS"]

    def test_outside_untouched(self) -> None:
        assert repair_iob2(["O", "O"]) == ["O", "O"]

    def test_i_after_o_promoted(self) -> None:
        assert repair_iob2(["O", "I-GEO"]) == ["O", "B-GEO"]
