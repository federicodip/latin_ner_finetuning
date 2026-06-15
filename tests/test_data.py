"""TDD: stratified split, LASLA remap, and file I/O for data prep."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from latin_ner.conll import Sentence
from latin_ner.data import (
    entity_types,
    is_multiword_range_row,
    load_crf_file,
    load_lasla_gold,
    normalize_lasla_label,
    run,
    stratified_sentence_split,
    stratify_key,
    write_jsonl,
)


def _s(token: str, labels: list[str]) -> Sentence:
    return Sentence(tokens=[token] * len(labels), labels=labels)


class TestEntityTypes:
    def test_extracts_types(self) -> None:
        assert entity_types(_s("a", ["B-PRS", "I-PRS", "O", "B-GEO"])) == {"PRS", "GEO"}

    def test_empty_when_all_o(self) -> None:
        assert entity_types(_s("a", ["O", "O"])) == set()


class TestStratifyKey:
    def test_none_when_no_entities(self) -> None:
        assert stratify_key(_s("a", ["O"])) == "none"

    def test_single_type(self) -> None:
        assert stratify_key(_s("a", ["B-PRS"])) == "PRS"

    def test_rarest_type_wins(self) -> None:
        # GEO is rarest in the corpus -> it is the stratum key.
        assert stratify_key(_s("a", ["B-PRS", "B-GEO"])) == "GEO"

    def test_multitoken_grp_is_own_stratum(self) -> None:
        assert stratify_key(Sentence(["a", "b"], ["B-GRP", "I-GRP"])) == "rare:I-GRP"


class TestStratifiedSplit:
    def _corpus(self, n: int) -> list[Sentence]:
        return [Sentence([str(i)], ["O"]) for i in range(n)]

    def test_partition_no_loss_no_duplication(self) -> None:
        sp = stratified_sentence_split(self._corpus(100), seed=1)
        seen = [s.tokens[0] for part in sp.values() for s in part]
        assert sorted(seen, key=int) == [str(i) for i in range(100)]
        assert len(seen) == 100

    def test_ratios_are_approximately_75_125_125(self) -> None:
        sp = stratified_sentence_split(self._corpus(1000), seed=1)
        assert 0.70 < len(sp["train"]) / 1000 < 0.80
        assert 0.09 < len(sp["dev"]) / 1000 < 0.16
        assert 0.09 < len(sp["test"]) / 1000 < 0.16

    def test_deterministic_for_same_seed(self) -> None:
        a = stratified_sentence_split(self._corpus(50), seed=7)
        b = stratified_sentence_split(self._corpus(50), seed=7)
        assert [s.tokens for s in a["test"]] == [s.tokens for s in b["test"]]

    def test_different_seed_changes_split(self) -> None:
        a = stratified_sentence_split(self._corpus(60), seed=1)
        b = stratified_sentence_split(self._corpus(60), seed=2)
        assert [s.tokens for s in a["test"]] != [s.tokens for s in b["test"]]

    def test_rare_stratum_lands_in_train(self) -> None:
        rare = [Sentence(["a", "b"], ["B-GRP", "I-GRP"]) for _ in range(8)]
        sp = stratified_sentence_split(rare + self._corpus(50), seed=3)
        assert any("I-GRP" in s.labels for s in sp["train"])


class TestLaslaHelpers:
    def test_multiword_range_row_detected(self) -> None:
        assert is_multiword_range_row("1-2") is True
        assert is_multiword_range_row("3") is False

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("B-PER", "B-PRS"),
            ("I-PER", "I-PRS"),
            ("B-PERS", "B-PRS"),
            ("B-LOC", "B-GEO"),
            ("B-GEO", "B-GEO"),
            ("B-GRP", "B-GRP"),
            ("I-GRP", "I-GRP"),
            ("B-DATE", "O"),  # DATE not in our label set -> dropped to O
            ("O", "O"),
        ],
    )
    def test_label_remap(self, raw: str, expected: str) -> None:
        assert normalize_lasla_label(raw) == expected

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_lasla_label("B-MISC")

    def test_malformed_label_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_lasla_label("PRS")  # no B-/I- prefix

    def test_header_row_is_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "lasla.tsv"
        p.write_text(
            "token_id\tform\tlemma\turi\tBIO_gold\n2\tCordi\tL\tURI\tB-PER\n",
            encoding="utf-8",
        )
        sents = load_lasla_gold(p)
        assert [t for s in sents for t in s.tokens] == ["Cordi"]


class TestFileIO:
    def test_load_crf_file_round_trip(self, tmp_path: Path) -> None:
        p = tmp_path / "x.crf"
        p.write_text("GEO-B\tGallia\n0\test\n\nPRS-B\tCicero\n", encoding="utf-8")
        sents = load_crf_file(p)
        assert sents[0].tokens == ["Gallia", "est"]
        assert sents[0].labels == ["B-GEO", "O"]
        assert sents[1].labels == ["B-PRS"]

    def test_load_crf_file_repairs_iob2(self, tmp_path: Path) -> None:
        # stray I- with no preceding same-type B- must be repaired to B-.
        p = tmp_path / "x.crf"
        p.write_text("GEO-I\tRoma\n", encoding="utf-8")
        assert load_crf_file(p)[0].labels == ["B-GEO"]

    def test_write_jsonl(self, tmp_path: Path) -> None:
        out = tmp_path / "train.jsonl"
        write_jsonl([Sentence(["a", "b"], ["O", "B-PRS"])], out)
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        assert rows == [{"tokens": ["a", "b"], "labels": ["O", "B-PRS"]}]

    def test_load_lasla_gold_skips_ranges_and_remaps(self, tmp_path: Path) -> None:
        p = tmp_path / "lasla.tsv"
        p.write_text(
            "1\tsemper\tL\tURI\tO\n"
            "1-2\tnumquamne\t\t\tO\n"  # MWT range row -> skipped
            "1\tnumquam\tL\tURI\tO\n"
            "2\tCordi\tL\tURI\tB-PER\n",  # PER -> PRS
            encoding="utf-8",
        )
        sents = load_lasla_gold(p)
        flat_tokens = [t for s in sents for t in s.tokens]
        flat_labels = [lab for s in sents for lab in s.labels]
        assert "numquamne" not in flat_tokens
        assert flat_labels[-1] == "B-PRS"


class TestRunIntegration:
    def _write_corpus(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        # Enough sentences that every split is non-empty.
        prose = "".join(f"PRS-B\tName{i}\n0\tword\n\n" for i in range(40))
        for name in ("GWtrain.crf", "GWtest.crf", "CW.crf", "PlinyElder.crf", "PlinyYounger.crf"):
            (d / name).write_text(prose, encoding="utf-8")
        (d / "Ovid.crf").write_text("GEO-B\tRoma\n0\tamat\n\n", encoding="utf-8")

    def test_run_writes_all_splits(self, tmp_path: Path) -> None:
        herod = tmp_path / "Annotation_1-1-19"
        self._write_corpus(herod)
        out = tmp_path / "out"
        manifest = run(herodotos_dir=herod, out_dir=out, seed=13)
        for split in ("train", "dev", "test", "poetry"):
            assert (out / f"{split}.jsonl").exists()
            assert manifest[split] > 0
        assert manifest["train"] > manifest["dev"]  # 75% vs 12.5%

    def test_run_with_lasla_writes_split(self, tmp_path: Path) -> None:
        herod = tmp_path / "Annotation_1-1-19"
        self._write_corpus(herod)
        lasla = tmp_path / "Latin_Gold_Data"
        lasla.mkdir()
        (lasla / "Juvenal_GOLD.tsv").write_text(
            "1\tsemper\tL\tURI\tO\n2\tCordi\tL\tURI\tB-PER\n", encoding="utf-8"
        )
        out = tmp_path / "out"
        manifest = run(herodotos_dir=herod, out_dir=out, lasla_dir=lasla, seed=13)
        assert (out / "lasla.jsonl").exists()
        assert manifest["lasla"] == 1

    def test_run_uses_explicit_data_git_sha(self, tmp_path: Path) -> None:
        # The container has no git, so the sha is captured on the host and passed
        # in; it must land verbatim in the manifest (reproducibility stamp).
        herod = tmp_path / "Annotation_1-1-19"
        self._write_corpus(herod)
        manifest = run(herodotos_dir=herod, out_dir=tmp_path / "o", data_git_sha="deadbeef")
        assert manifest["data_git_sha"] == "deadbeef"

    def test_run_is_deterministic(self, tmp_path: Path) -> None:
        herod = tmp_path / "Annotation_1-1-19"
        self._write_corpus(herod)
        m1 = run(herodotos_dir=herod, out_dir=tmp_path / "a", seed=13)
        m2 = run(herodotos_dir=herod, out_dir=tmp_path / "b", seed=13)
        assert (tmp_path / "a" / "test.jsonl").read_text(encoding="utf-8") == (
            tmp_path / "b" / "test.jsonl"
        ).read_text(encoding="utf-8")
        assert m1 == m2
