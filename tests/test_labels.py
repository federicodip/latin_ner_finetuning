"""TDD: BIO label <-> id encoding for the 7-label PRS/GEO/GRP set."""

import pytest

from latin_ner import labels


def test_num_labels_is_seven() -> None:
    assert labels.NUM_LABELS == 7
    assert len(labels.LABELS) == 7


def test_label_order_matches_output_contract() -> None:
    # Order is part of the OUTPUT CONTRACT (id2label in config.json).
    assert labels.LABELS == [
        "O",
        "B-PRS",
        "I-PRS",
        "B-GEO",
        "I-GEO",
        "B-GRP",
        "I-GRP",
    ]


def test_o_is_zero() -> None:
    assert labels.LABEL2ID["O"] == 0
    assert labels.ID2LABEL[0] == "O"


def test_label2id_id2label_are_consistent() -> None:
    for idx, lab in labels.ID2LABEL.items():
        assert labels.LABEL2ID[lab] == idx
    assert set(labels.LABEL2ID.values()) == set(range(labels.NUM_LABELS))


def test_entity_types() -> None:
    assert labels.ENTITY_TYPES == ("PRS", "GEO", "GRP")


def test_encode_decode_roundtrip() -> None:
    seq = ["O", "B-PRS", "I-PRS", "B-GEO", "O", "B-GRP", "I-GRP"]
    ids = labels.encode(seq)
    assert ids == [0, 1, 2, 3, 0, 5, 6]
    assert labels.decode(ids) == seq


def test_label_to_id_unknown_raises() -> None:
    with pytest.raises(KeyError):
        labels.label_to_id("B-DATE")


def test_id_to_label_out_of_range_raises() -> None:
    with pytest.raises(KeyError):
        labels.id_to_label(99)
