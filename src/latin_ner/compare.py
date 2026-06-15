"""Direct head-to-head: run LatinCy (spaCy) on our exact splits, score with our
seqeval scorer, and tabulate the delta vs our fine-tuned model.

LatinCy is fed our gold-tokenized words via ``Doc(nlp.vocab, words=...)`` so token
boundaries match the gold exactly (no tokenization confound); its ``doc.ents`` are
converted to IOB2 over those tokens and scored with :mod:`latin_ner.evaluate`.

The pure pieces (label map, ents->BIO, comparison table) are unit-tested. The
spaCy load/inference + ``main`` are cluster glue (run in the conda compare
container).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .labels import ENTITY_TYPES

#: LatinCy's general NER scheme -> our native types (verified from its model card).
LATINCY_LABEL_MAP: dict[str, str] = {"PERSON": "PRS", "LOC": "GEO", "NORP": "GRP"}

#: Splits to compare. LASLA is excluded — it is in LatinCy's training data, so a
#: LatinCy score there would be contaminated/unfair.
COMPARE_SPLITS: dict[str, str] = {"in_domain_test": "test.jsonl", "poetry": "poetry.jsonl"}


def spacy_ents_to_bio(
    ent_spans: Sequence[tuple[int, int, str]],
    n_tokens: int,
    label_map: Mapping[str, str],
) -> list[str]:
    """Convert spaCy entity spans ``(start_tok, end_tok, label)`` into an IOB2
    sequence over ``n_tokens``. Labels absent from ``label_map`` are dropped;
    out-of-range spans are ignored."""
    labels = ["O"] * n_tokens
    for start, end, spacy_label in ent_spans:
        mapped = label_map.get(spacy_label)
        if mapped is None or start < 0 or end > n_tokens or start >= end:
            continue
        labels[start] = f"B-{mapped}"
        for i in range(start + 1, end):
            labels[i] = f"I-{mapped}"
    return labels


def gold_token_char_spans(words: Sequence[str]) -> list[tuple[int, int]]:
    """Char spans of each word in spaCy's space-joined Doc text (``"w1 w2 ... "``)."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for w in words:
        spans.append((pos, pos + len(w)))
        pos += len(w) + 1  # +1 for the space spaCy puts after each token
    return spans


def char_ents_to_bio(
    words: Sequence[str],
    ent_char_spans: Sequence[tuple[int, int, str]],
    label_map: Mapping[str, str],
) -> list[str]:
    """Map entity CHARACTER spans onto gold tokens by overlap -> IOB2.

    Robust to LatinCy pipeline components that **retokenize** (e.g.
    ``enclitic_splitter`` splitting ``-que``): char offsets index the unchanged
    Doc text, so a changed token count can't misalign the labels (token-index
    mapping would silently break and score LatinCy unfairly).
    """
    tok_spans = gold_token_char_spans(words)
    labels = ["O"] * len(words)
    for cs, ce, spacy_label in ent_char_spans:
        mapped = label_map.get(spacy_label)
        if mapped is None:
            continue
        covered = [i for i, (ts, te) in enumerate(tok_spans) if ts < ce and cs < te]
        for rank, i in enumerate(covered):
            labels[i] = f"{'B' if rank == 0 else 'I'}-{mapped}"
    return labels


def build_comparison(
    our_splits: Mapping[str, Any],
    latincy_scores: Mapping[str, Any],
    model_name: str,
) -> dict[str, Any]:
    """Build the comparison structure (our model vs LatinCy) over the splits
    present in both, with strict macro + per-type deltas."""
    splits: dict[str, Any] = {}
    for name in latincy_scores:
        if name not in our_splits:
            continue
        ours = our_splits[name]["strict"]
        theirs = latincy_scores[name]["strict"]
        per_type = {}
        for t in ENTITY_TYPES:
            ours_f1 = our_splits[name]["strict"]["per_type"].get(t, {}).get("f1", 0.0)
            lat_f1 = latincy_scores[name]["strict"]["per_type"].get(t, {}).get("f1", 0.0)
            per_type[t] = {"our": ours_f1, "latincy": lat_f1, "delta": ours_f1 - lat_f1}
        splits[name] = {
            "our_strict_macro": ours["macro_f1"],
            "latincy_strict_macro": theirs["macro_f1"],
            "delta_strict_macro": ours["macro_f1"] - theirs["macro_f1"],
            "our_relaxed_macro": our_splits[name]["relaxed"]["macro_f1"],
            "latincy_relaxed_macro": latincy_scores[name]["relaxed"]["macro_f1"],
            "per_type": per_type,
        }
    return {"latincy_model": model_name, "splits": splits}


def render_comparison_md(comparison: Mapping[str, Any]) -> str:
    """Render a grep-able markdown comparison table."""
    out: list[str] = [
        f"# Direct comparison: fine-tuned LatinBERT vs LatinCy `{comparison['latincy_model']}`",
        "",
        "Entity-level strict F1 (seqeval IOB2), both scored on the identical gold-tokenized",
        "split. LASLA excluded (in LatinCy's training data).",
        "",
    ]
    for name, s in comparison["splits"].items():
        out.append(f"## Split: {name}")
        out.append("| class | ours | LatinCy | delta |")
        out.append("|---|---|---|---|")
        for t in ENTITY_TYPES:
            pt = s["per_type"][t]
            out.append(f"| {t} | {pt['our']:.3f} | {pt['latincy']:.3f} | {pt['delta']:+.3f} |")
        out.append(
            f"| **macro (strict)** | **{s['our_strict_macro']:.3f}** | "
            f"**{s['latincy_strict_macro']:.3f}** | **{s['delta_strict_macro']:+.3f}** |"
        )
        rdelta = s["our_relaxed_macro"] - s["latincy_relaxed_macro"]
        out.append(
            f"| macro (relaxed) | {s['our_relaxed_macro']:.3f} | "
            f"{s['latincy_relaxed_macro']:.3f} | {rdelta:+.3f} |"
        )
        out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Cluster glue (lazy spaCy import; runs in the conda compare container)
# --------------------------------------------------------------------------- #
def latincy_predict_split(
    nlp: Any, sentences: Sequence[Sequence[str]]
) -> list[list[str]]:  # pragma: no cover
    """Run LatinCy on gold-tokenized sentences -> IOB2 over the same tokens."""
    from spacy.tokens import Doc

    preds: list[list[str]] = []
    for words in sentences:
        doc = Doc(nlp.vocab, words=list(words))
        for _, proc in nlp.pipeline:
            doc = proc(doc)
        # Use CHARACTER offsets (not token indices): LatinCy's enclitic_splitter
        # can retokenize, which would break token-index alignment with the gold.
        ent_char_spans = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
        preds.append(char_ents_to_bio(words, ent_char_spans, LATINCY_LABEL_MAP))
    return preds


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover
    import spacy

    from .evaluate import evaluate_split, load_jsonl

    parser = argparse.ArgumentParser(
        description="Score LatinCy on our splits; compare to our model."
    )
    parser.add_argument("--model", default="la_core_web_lg", help="spaCy/LatinCy model name")
    parser.add_argument("--data-dir", required=True, help="dir with test/poetry .jsonl")
    parser.add_argument("--our-eval", default="eval/latin_ner_eval.json")
    parser.add_argument("--out-json", default="eval/latincy_comparison_direct.json")
    parser.add_argument("--out-md", default="eval/latincy_comparison_direct.md")
    args = parser.parse_args(argv)

    nlp = spacy.load(args.model)
    data_dir = Path(args.data_dir)
    latincy_scores: dict[str, Any] = {}
    for name, fname in COMPARE_SPLITS.items():
        fp = data_dir / fname
        if not fp.exists():
            continue
        tokens, gold = load_jsonl(fp)
        pred = latincy_predict_split(nlp, tokens)
        latincy_scores[name] = evaluate_split(name, gold, pred)

    if not latincy_scores:
        raise SystemExit(
            f"No split files found in {data_dir} (expected {list(COMPARE_SPLITS.values())}); "
            "did prepare_data run and is --data-dir correct?"
        )

    our_eval = json.loads(Path(args.our_eval).read_text(encoding="utf-8"))
    comparison = build_comparison(our_eval["splits"], latincy_scores, args.model)
    comparison["latincy_raw_scores"] = latincy_scores

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    Path(args.out_md).write_text(render_comparison_md(comparison), encoding="utf-8")
    print(render_comparison_md(comparison))


if __name__ == "__main__":  # pragma: no cover
    main()
