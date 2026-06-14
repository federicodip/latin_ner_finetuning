"""Data prep: load Herodotos ``.crf``, stratified split, optional LASLA eval.

The *network* step (``git clone`` of the AGPL Herodotos repo onto /scratch)
lives in ``jobs/prepare_data.sh``. Everything here reads already-present files,
so it is fully unit-testable offline.

In-domain corpus = the 4 prose works (Caesar BG split across two files, Caesar
BC, Pliny Elder, Pliny Younger), pooled and split 75/12.5/12.5 at the sentence
level with a fixed seed, stratified so rare types reach every split. Ovid (Ars
Amatoria) is held out whole as the poetry / out-of-domain test.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .conll import Sentence, parse_crf, repair_iob2
from .spans import bio_to_spans

#: Prose works -> pooled in-domain set (Ovid is poetry, kept separate).
PROSE_FILES: tuple[str, ...] = (
    "GWtrain.crf",
    "GWtest.crf",
    "CW.crf",
    "PlinyElder.crf",
    "PlinyYounger.crf",
)
POETRY_FILE: str = "Ovid.crf"

#: Rarity order (rarest first) for choosing a sentence's stratum.
_RARITY: tuple[str, ...] = ("GEO", "GRP", "PRS")

#: LASLA type aliases -> our native types; DATE is dropped to O.
_LASLA_TYPE_MAP: dict[str, str] = {
    "PER": "PRS",
    "PERS": "PRS",
    "PRS": "PRS",
    "LOC": "GEO",
    "GEO": "GEO",
    "GRP": "GRP",
}
_LASLA_DROP: frozenset[str] = frozenset({"DATE"})
_LASLA_HEADER_SENTINELS: frozenset[str] = frozenset({"BIO_gold", "label", "BIO"})


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def entity_types(sentence: Sentence) -> set[str]:
    """Set of entity types present in a sentence's labels."""
    return {lab.split("-", 1)[1] for lab in sentence.labels if lab != "O"}


def stratify_key(sentence: Sentence) -> str:
    """A stratification bucket key: ultra-rare multi-token spans get their own
    bucket; otherwise the rarest single type present; else ``"none"``."""
    labels = sentence.labels
    if "I-GRP" in labels:
        return "rare:I-GRP"
    if "I-GEO" in labels:
        return "rare:I-GEO"
    types = entity_types(sentence)
    if not types:
        return "none"
    for t in _RARITY:
        if t in types:
            return t
    return "none"


def stratified_sentence_split(
    sentences: Sequence[Sentence],
    *,
    seed: int = 13,
    ratios: tuple[float, float, float] = (0.75, 0.125, 0.125),
) -> dict[str, list[Sentence]]:
    """Deterministic stratified 75/12.5/12.5 sentence-level split."""
    import random

    buckets: dict[str, list[int]] = {}
    for i, s in enumerate(sentences):
        buckets.setdefault(stratify_key(s), []).append(i)

    rng = random.Random(seed)
    train: list[int] = []
    dev: list[int] = []
    test: list[int] = []
    for key in sorted(buckets):
        idxs = buckets[key][:]
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = round(n * ratios[0])
        n_dev = round(n * ratios[1])
        train += idxs[:n_train]
        dev += idxs[n_train : n_train + n_dev]
        test += idxs[n_train + n_dev :]

    return {
        "train": [sentences[i] for i in sorted(train)],
        "dev": [sentences[i] for i in sorted(dev)],
        "test": [sentences[i] for i in sorted(test)],
    }


def is_multiword_range_row(token_id: str) -> bool:
    """LASLA/CoNLL-U multiword-token range rows have ids like ``1-2``."""
    return "-" in token_id


def normalize_lasla_label(label: str) -> str:
    """Remap a LASLA BIO label to our native PRS/GEO/GRP set.

    PER/PERS -> PRS, LOC -> GEO, DATE -> O. Unknown types raise ``ValueError``.
    """
    s = label.strip()
    if s in {"O", "0"}:
        return "O"
    bio, _, etype = s.partition("-")
    if bio not in {"B", "I"} or not etype:
        raise ValueError(f"Malformed LASLA label: {label!r}")
    if etype in _LASLA_DROP:
        return "O"
    if etype in _LASLA_TYPE_MAP:
        return f"{bio}-{_LASLA_TYPE_MAP[etype]}"
    raise ValueError(f"Unknown LASLA entity type: {label!r}")


# --------------------------------------------------------------------------- #
# File I/O
# --------------------------------------------------------------------------- #
def load_crf_file(path: Path | str) -> list[Sentence]:
    """Parse a Herodotos ``.crf`` file, repairing IOB2 per sentence."""
    text = Path(path).read_text(encoding="utf-8")
    return [Sentence(tokens=s.tokens, labels=repair_iob2(s.labels)) for s in parse_crf(text)]


def load_lasla_gold(path: Path | str) -> list[Sentence]:
    """Parse a LASLA GOLD ``.tsv``: skip ``#`` comments, header rows, and MWT
    range rows; remap labels to our native set; repair IOB2."""
    sentences: list[Sentence] = []
    tokens: list[str] = []
    labels: list[str] = []

    def flush() -> None:
        if tokens:
            sentences.append(Sentence(tokens=tokens.copy(), labels=repair_iob2(labels.copy())))
            tokens.clear()
            labels.clear()

    for raw in Path(path).read_text(encoding="utf-8").split("\n"):
        line = raw.rstrip("\r")
        if not line.strip():
            flush()
            continue
        if line.startswith("#"):
            continue
        parts = line.split("\t")
        if (
            len(parts) < 5
            or is_multiword_range_row(parts[0])
            or parts[4] in _LASLA_HEADER_SENTINELS
        ):
            continue
        tokens.append(parts[1])
        labels.append(normalize_lasla_label(parts[4]))
    flush()
    return sentences


def write_jsonl(sentences: Sequence[Sentence], path: Path | str) -> None:
    """Write sentences as JSONL: ``{"tokens": [...], "labels": [...]}``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for s in sentences:
            f.write(json.dumps({"tokens": s.tokens, "labels": s.labels}, ensure_ascii=False) + "\n")


def _git_sha(repo_dir: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _entity_counts(sentences: Sequence[Sentence]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in sentences:
        for span in bio_to_spans(s.labels):
            counts[span.etype] = counts.get(span.etype, 0) + 1
    return dict(sorted(counts.items()))


def run(
    *,
    herodotos_dir: Path | str,
    out_dir: Path | str,
    lasla_dir: Path | str | None = None,
    seed: int = 13,
) -> dict[str, object]:
    """Parse, split, and write JSONL splits; return a manifest dict."""
    herodotos_dir = Path(herodotos_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prose: list[Sentence] = []
    for name in PROSE_FILES:
        fp = herodotos_dir / name
        if fp.exists():
            prose += load_crf_file(fp)

    splits = stratified_sentence_split(prose, seed=seed)
    poetry_fp = herodotos_dir / POETRY_FILE
    poetry = load_crf_file(poetry_fp) if poetry_fp.exists() else []

    write_jsonl(splits["train"], out_dir / "train.jsonl")
    write_jsonl(splits["dev"], out_dir / "dev.jsonl")
    write_jsonl(splits["test"], out_dir / "test.jsonl")
    write_jsonl(poetry, out_dir / "poetry.jsonl")

    manifest: dict[str, object] = {
        "seed": seed,
        "train": len(splits["train"]),
        "dev": len(splits["dev"]),
        "test": len(splits["test"]),
        "poetry": len(poetry),
        "data_git_sha": _git_sha(herodotos_dir),
        "entity_counts": _entity_counts(prose),
    }

    if lasla_dir is not None:
        lasla: list[Sentence] = []
        for fp in sorted(Path(lasla_dir).glob("*GOLD.tsv")):
            lasla += load_lasla_gold(fp)
        write_jsonl(lasla, out_dir / "lasla.jsonl")
        manifest["lasla"] = len(lasla)

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Prepare Herodotos Latin NER splits.")
    parser.add_argument("--herodotos-dir", required=True, help="Path to Annotation_1-1-19/")
    parser.add_argument("--out-dir", required=True, help="Where to write *.jsonl + manifest.json")
    parser.add_argument(
        "--lasla-dir", default=None, help="Optional Latin_Gold_Data/ for LASLA eval"
    )
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)
    manifest = run(
        herodotos_dir=args.herodotos_dir,
        out_dir=args.out_dir,
        lasla_dir=args.lasla_dir,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
