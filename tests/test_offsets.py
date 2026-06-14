"""TDD: whitespace word -> character-span tracking for offset emission."""

from __future__ import annotations

from latin_ner.offsets import spans_to_char_offsets, whitespace_word_spans


class TestWhitespaceWordSpans:
    def test_basic(self) -> None:
        assert whitespace_word_spans("Gallia est") == [
            ("Gallia", 0, 6),
            ("est", 7, 10),
        ]

    def test_offsets_slice_back_to_word(self) -> None:
        text = "  Gallia   est\tomnis\n"
        for word, start, end in whitespace_word_spans(text):
            assert text[start:end] == word

    def test_empty_and_whitespace_only(self) -> None:
        assert whitespace_word_spans("") == []
        assert whitespace_word_spans("   \t\n ") == []

    def test_punctuation_attached(self) -> None:
        # whitespace tokenization keeps trailing punctuation on the word
        assert whitespace_word_spans("Roma, Italia") == [
            ("Roma,", 0, 5),
            ("Italia", 6, 12),
        ]


class TestSpansToCharOffsets:
    def test_maps_token_span_to_char_span(self) -> None:
        words = whitespace_word_spans("Marcus Tullius Cicero venit")
        # token span [0,3) = "Marcus Tullius Cicero", type PRS
        out = spans_to_char_offsets("Marcus Tullius Cicero venit", words, [(0, 3, "PRS")])
        assert out == [{"text": "Marcus Tullius Cicero", "start": 0, "end": 21, "type": "PRS"}]

    def test_single_token_entity(self) -> None:
        text = "in Gallia"
        words = whitespace_word_spans(text)
        out = spans_to_char_offsets(text, words, [(1, 2, "GEO")])
        assert out == [{"text": "Gallia", "start": 3, "end": 9, "type": "GEO"}]

    def test_empty_spans(self) -> None:
        text = "nulla"
        assert spans_to_char_offsets(text, whitespace_word_spans(text), []) == []
