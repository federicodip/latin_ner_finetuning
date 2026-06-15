"""Entity-level evaluation: strict (seqeval, IOB2) + relaxed, per-class.

Strict scoring uses seqeval ``mode="strict", scheme="IOB2"`` — NEVER the default
seqeval mode, which scores broken spans 1.0. Relaxed scoring (token-overlap,
same-type) comes from :mod:`latin_ner.spans`.

The pure scoring/report/I-O functions here are unit-tested offline. Model
inference (``main``) lazily imports torch/transformers and runs on the cluster.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

from .labels import ENTITY_TYPES
from .spans import bio_to_spans, score_relaxed

#: Breakage FLOOR on in-domain strict macro-F1 — NOT the quality target. The
#: paper's 0.88-0.90 is on an unreproducible split (seed unreleased) and
#: cross-split NER variance is ~0.02-0.03, so we gate only on "clearly learned"
#: and report the actual number against the 0.88-0.90 target for human judgement.
SANITY_MACRO_F1: float = 0.83

LabelSeqs = Sequence[Sequence[str]]


def _f(x: Any) -> float:
    return float(x)


def _i(x: Any) -> int:
    return int(x)


def _macro_over_present(per_type: dict[str, dict[str, Any]]) -> float:
    """Macro-F1 over classes that have gold support (>0). Matches seqeval's
    'macro avg' and penalizes a class the model never recalls."""
    present = [t for t in ENTITY_TYPES if per_type[t]["support"] > 0]
    if not present:
        return 0.0
    return float(sum(per_type[t]["f1"] for t in present) / len(present))


def strict_scores(gold: LabelSeqs, pred: LabelSeqs) -> dict[str, Any]:
    """Strict entity-level P/R/F1 via seqeval (IOB2), per-class + micro + macro."""
    rep = classification_report(
        list(gold),
        list(pred),
        mode="strict",
        scheme=IOB2,
        output_dict=True,
        zero_division=0,
    )
    per_type: dict[str, dict[str, Any]] = {}
    for t in ENTITY_TYPES:
        d = rep.get(t, {"precision": 0, "recall": 0, "f1-score": 0, "support": 0})
        per_type[t] = {
            "precision": _f(d["precision"]),
            "recall": _f(d["recall"]),
            "f1": _f(d["f1-score"]),
            "support": _i(d["support"]),
        }
    micro = rep.get("micro avg", {"precision": 0, "recall": 0, "f1-score": 0, "support": 0})
    return {
        "macro_f1": _macro_over_present(per_type),
        "micro": {
            "precision": _f(micro["precision"]),
            "recall": _f(micro["recall"]),
            "f1": _f(micro["f1-score"]),
            "support": _i(micro["support"]),
        },
        "per_type": per_type,
    }


def relaxed_scores(gold: LabelSeqs, pred: LabelSeqs) -> dict[str, Any]:
    """Relaxed entity-level P/R/F1 (token-overlap, same-type), per-class."""
    sr = score_relaxed(gold, pred)
    per_type = {
        t: {
            "precision": sr.per_type[t].precision,
            "recall": sr.per_type[t].recall,
            "f1": sr.per_type[t].f1,
            "support": sr.per_type[t].support,
        }
        for t in ENTITY_TYPES
    }
    return {
        "macro_f1": _macro_over_present(per_type),
        "micro": {
            "precision": sr.micro.precision,
            "recall": sr.micro.recall,
            "f1": sr.micro.f1,
            "support": sr.micro.support,
        },
        "per_type": per_type,
    }


def evaluate_split(name: str, gold: LabelSeqs, pred: LabelSeqs) -> dict[str, Any]:
    """Score one split, returning strict + relaxed metrics and entity counts."""
    if len(gold) != len(pred):
        raise ValueError(f"{name}: gold/pred sentence count differs")
    return {
        "name": name,
        "n_sentences": len(gold),
        "n_gold_entities": sum(len(bio_to_spans(s)) for s in gold),
        "n_pred_entities": sum(len(bio_to_spans(s)) for s in pred),
        "strict": strict_scores(gold, pred),
        "relaxed": relaxed_scores(gold, pred),
    }


def build_report(splits: Sequence[dict[str, Any]], *, repro: dict[str, Any]) -> dict[str, Any]:
    """Assemble the full report dict, keyed by split name, with an acceptance
    gate on the in-domain test split's strict macro-F1."""
    by_name = {s["name"]: s for s in splits}
    in_dom = by_name.get("in_domain_test")
    macro = in_dom["strict"]["macro_f1"] if in_dom else None
    return {
        "repro": repro,
        "acceptance": {
            "metric": "in_domain_test strict macro-F1 (floor; paper target 0.88-0.90)",
            "threshold": SANITY_MACRO_F1,
            "in_domain_macro_f1": macro,
            "passes": macro is not None and macro >= SANITY_MACRO_F1,
        },
        "splits": by_name,
    }


def _fmt_block(title: str, scores: dict[str, Any]) -> list[str]:
    micro = scores["micro"]
    lines = [
        f"{title:7s} macro-F1={scores['macro_f1']:.4f}  "
        f"micro P={micro['precision']:.4f} R={micro['recall']:.4f} F1={micro['f1']:.4f}"
    ]
    for t in ENTITY_TYPES:
        d = scores["per_type"][t]
        lines.append(
            f"  {t}  P={d['precision']:.3f} R={d['recall']:.3f} "
            f"F1={d['f1']:.3f} support={d['support']}"
        )
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    """Render a grep-able markdown report."""
    out: list[str] = ["# Latin NER evaluation", ""]
    for k, v in report["repro"].items():
        out.append(f"- {k}: {v}")
    acc = report["acceptance"]
    verdict = "PASS" if acc["passes"] else "CHECK"
    out += [
        "",
        "## Acceptance",
        f"{acc['metric']} = {acc['in_domain_macro_f1']} "
        f"(threshold {acc['threshold']}) -> {verdict}",
        "",
    ]
    for name, split in report["splits"].items():
        out.append(
            f"## Split: {name}  (sentences={split['n_sentences']}, "
            f"gold_entities={split['n_gold_entities']}, pred_entities={split['n_pred_entities']})"
        )
        out += _fmt_block("STRICT", split["strict"])
        out += _fmt_block("RELAXED", split["relaxed"])
        out.append("")
    return "\n".join(out)


def write_eval_outputs(report: dict[str, Any], json_path: Path | str, md_path: Path | str) -> None:
    """Write the report JSON and the markdown summary."""
    jp, mp = Path(json_path), Path(md_path)
    jp.parent.mkdir(parents=True, exist_ok=True)
    mp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    mp.write_text(render_markdown(report), encoding="utf-8")


def load_jsonl(path: Path | str) -> tuple[list[list[str]], list[list[str]]]:
    """Load a ``{"tokens": [...], "labels": [...]}`` JSONL into parallel lists."""
    tokens: list[list[str]] = []
    labels: list[list[str]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tokens.append(row["tokens"])
        labels.append(row["labels"])
    return tokens, labels


# --------------------------------------------------------------------------- #
# Cluster inference (lazy torch import; not unit-tested)
# --------------------------------------------------------------------------- #
def _predict_split(  # pragma: no cover
    model: Any,
    tokenizer: Any,
    sentences: list[list[str]],
    *,
    max_length: int,
    device: str,
    batch_size: int,
) -> list[list[str]]:
    import torch

    from .align import decode_predictions
    from .labels import ID2LABEL

    preds: list[list[str]] = []
    model.eval()
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        enc = tokenizer(
            batch,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        with torch.no_grad():
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
        pred_ids = logits.argmax(-1).cpu().tolist()
        for b, words in enumerate(batch):
            word_ids = enc.word_ids(batch_index=b)
            word_pred_ids = decode_predictions(word_ids, pred_ids[b])
            seq = [ID2LABEL[p] for p in word_pred_ids]
            # Re-align to gold length: words truncated past max_length -> "O".
            if len(seq) < len(words):
                seq += ["O"] * (len(words) - len(seq))
            preds.append(seq[: len(words)])
    return preds


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover
    import torch
    import transformers
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    parser = argparse.ArgumentParser(description="Evaluate a Latin NER checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True, help="dir with test/poetry/lasla .jsonl")
    parser.add_argument("--out-json", default="eval/latin_ner_eval.json")
    parser.add_argument("--out-md", default="eval/latin_ner_eval.md")
    parser.add_argument("--max-length", type=int, default=512)  # match training; avoid truncation
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--git-sha", default="unknown", help="this repo's git sha (repro stamp)")
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Call through Any-typed aliases so mypy is happy regardless of whether
    # transformers' (cross-version) type stubs are present in the lint env.
    auto_model: Any = AutoModelForTokenClassification
    auto_tok: Any = AutoTokenizer
    model = auto_model.from_pretrained(args.checkpoint, trust_remote_code=True).to(device)
    # use_fast=True is REQUIRED: _predict_split relies on word_ids(), which only
    # fast tokenizers expose. Being explicit also makes a missing fast-tokenizer
    # source file fail loudly here rather than silently degrading to the slow one.
    tokenizer = auto_tok.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=True)

    data_dir = Path(args.data_dir)
    manifest = {}
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    split_files = {
        "in_domain_test": "test.jsonl",
        "poetry": "poetry.jsonl",
        "lasla_cross_genre": "lasla.jsonl",
    }
    splits = []
    for name, fname in split_files.items():
        fp = data_dir / fname
        if not fp.exists():
            continue
        tokens, gold = load_jsonl(fp)
        pred = _predict_split(
            model,
            tokenizer,
            tokens,
            max_length=args.max_length,
            device=device,
            batch_size=args.batch_size,
        )
        splits.append(evaluate_split(name, gold, pred))

    repro = {
        "backbone": getattr(model.config, "_name_or_path", "latincy/latin-bert"),
        "checkpoint": args.checkpoint,
        "transformers_version": transformers.__version__,
        "git_sha": args.git_sha,
        "data_git_sha": manifest.get("data_git_sha", "unknown"),
        "num_labels": model.config.num_labels,
    }
    report = build_report(splits, repro=repro)
    write_eval_outputs(report, args.out_json, args.out_md)
    print(render_markdown(report))


if __name__ == "__main__":  # pragma: no cover
    main()
