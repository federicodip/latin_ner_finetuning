"""TDD: LatinCy spaCy-ents -> BIO mapping and the comparison table."""

from __future__ import annotations

from latin_ner.compare import (
    LATINCY_LABEL_MAP,
    build_comparison,
    char_ents_to_bio,
    gold_token_char_spans,
    render_comparison_md,
    spacy_ents_to_bio,
)


class TestGoldTokenCharSpans:
    def test_spans_match_spacy_space_joined_text(self) -> None:
        # spaCy Doc(words=W) text is "w1 w2 ... wN " (each token + one space).
        assert gold_token_char_spans(["arma", "virum"]) == [(0, 4), (5, 10)]

    def test_empty(self) -> None:
        assert gold_token_char_spans([]) == []


class TestCharEntsToBio:
    def test_single_token_entity(self) -> None:
        # entity char span over "arma" -> first token is PRS
        spans = [(0, 4, "PERSON")]
        assert char_ents_to_bio(["arma", "virum"], spans, LATINCY_LABEL_MAP) == ["B-PRS", "O"]

    def test_multitoken_entity(self) -> None:
        spans = [(0, 10, "NORP")]  # covers both "arma" and "virum"
        assert char_ents_to_bio(["arma", "virum"], spans, LATINCY_LABEL_MAP) == ["B-GRP", "I-GRP"]

    def test_robust_to_retokenization(self) -> None:
        # LatinCy splits "armaque" -> ent only over the "arma" part (chars 0-4);
        # via char overlap the whole gold token "armaque" still gets the label.
        words = ["armaque", "virum"]  # gold token spans: (0,7), (8,13)
        spans = [(0, 4, "PERSON")]
        assert char_ents_to_bio(words, spans, LATINCY_LABEL_MAP) == ["B-PRS", "O"]

    def test_unmapped_label_dropped(self) -> None:
        assert char_ents_to_bio(["a", "b"], [(0, 1, "DATE")], LATINCY_LABEL_MAP) == ["O", "O"]

    def test_no_entities(self) -> None:
        assert char_ents_to_bio(["a", "b"], [], LATINCY_LABEL_MAP) == ["O", "O"]

    def test_entity_outside_tokens_ignored(self) -> None:
        assert char_ents_to_bio(["a", "b"], [(50, 60, "PERSON")], LATINCY_LABEL_MAP) == ["O", "O"]


class TestLabelMap:
    def test_maps_general_scheme_to_native(self) -> None:
        # LatinCy emits PERSON/LOC/NORP; map to our PRS/GEO/GRP.
        assert LATINCY_LABEL_MAP == {"PERSON": "PRS", "LOC": "GEO", "NORP": "GRP"}


class TestSpacyEntsToBio:
    def test_single_token_entity(self) -> None:
        assert spacy_ents_to_bio([(0, 1, "PERSON")], 3, LATINCY_LABEL_MAP) == ["B-PRS", "O", "O"]

    def test_multitoken_entity(self) -> None:
        assert spacy_ents_to_bio([(1, 3, "LOC")], 4, LATINCY_LABEL_MAP) == [
            "O",
            "B-GEO",
            "I-GEO",
            "O",
        ]

    def test_norp_maps_to_grp(self) -> None:
        assert spacy_ents_to_bio([(0, 1, "NORP")], 1, LATINCY_LABEL_MAP) == ["B-GRP"]

    def test_unmapped_label_is_dropped(self) -> None:
        # spaCy may emit DATE/ORG/etc. that have no native equivalent -> treat as O.
        assert spacy_ents_to_bio([(0, 1, "DATE")], 2, LATINCY_LABEL_MAP) == ["O", "O"]

    def test_two_entities(self) -> None:
        out = spacy_ents_to_bio([(0, 1, "PERSON"), (2, 4, "NORP")], 4, LATINCY_LABEL_MAP)
        assert out == ["B-PRS", "O", "B-GRP", "I-GRP"]

    def test_no_entities(self) -> None:
        assert spacy_ents_to_bio([], 3, LATINCY_LABEL_MAP) == ["O", "O", "O"]

    def test_out_of_range_span_ignored(self) -> None:
        assert spacy_ents_to_bio([(2, 5, "PERSON")], 3, LATINCY_LABEL_MAP) == ["O", "O", "O"]


class TestBuildComparison:
    def _our(self) -> dict:
        return {
            "in_domain_test": {
                "strict": {
                    "macro_f1": 0.84,
                    "per_type": {
                        "PRS": {"f1": 0.88},
                        "GEO": {"f1": 0.79},
                        "GRP": {"f1": 0.85},
                    },
                },
                "relaxed": {"macro_f1": 0.87},
            }
        }

    def _latincy(self) -> dict:
        return {
            "in_domain_test": {
                "strict": {
                    "macro_f1": 0.43,
                    "per_type": {
                        "PRS": {"f1": 0.64},
                        "GEO": {"f1": 0.61},
                        "GRP": {"f1": 0.02},
                    },
                },
                "relaxed": {"macro_f1": 0.45},
            }
        }

    def test_delta_computed(self) -> None:
        c = build_comparison(self._our(), self._latincy(), "la_core_web_lg")
        s = c["splits"]["in_domain_test"]
        assert s["our_strict_macro"] == 0.84
        assert s["latincy_strict_macro"] == 0.43
        assert round(s["delta_strict_macro"], 2) == 0.41
        assert round(s["per_type"]["GRP"]["delta"], 2) == 0.83
        assert c["latincy_model"] == "la_core_web_lg"

    def test_only_shared_splits(self) -> None:
        # poetry present for our model but not scored for LatinCy -> excluded.
        our = {**self._our(), "poetry": {"strict": {"macro_f1": 0.49, "per_type": {}}}}
        c = build_comparison(our, self._latincy(), "la_core_web_lg")
        assert set(c["splits"]) == {"in_domain_test"}


class TestRenderMarkdown:
    def test_is_grepable(self) -> None:
        c = build_comparison(
            TestBuildComparison()._our(), TestBuildComparison()._latincy(), "la_core_web_lg"
        )
        md = render_comparison_md(c)
        assert "la_core_web_lg" in md
        assert "in_domain_test" in md
        assert "delta" in md.lower()
        assert "GRP" in md
