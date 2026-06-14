"""FRICTION GATE: prove LatinBERT loads HF-compatibly before training.

Run on the cluster (needs torch + transformers + internet/cache):

    python -m latin_ner.gate_check

Hard requirements (exit 1 on any failure):
  1. the custom LatinBERT fast tokenizer loads (trust_remote_code) and exposes
     word_ids();
  2. AutoModelForTokenClassification builds on the backbone with num_labels=7
     and the correct id2label;
  3. a forward pass returns logits shaped [1, L, 7];
  4. character offsets for a predicted span are emitted correctly.

It also reports (non-fatal) whether the tokenizer's own offset_mapping works.
"""

from __future__ import annotations

import sys
from typing import Any

from .labels import ID2LABEL, LABEL2ID, NUM_LABELS
from .offsets import spans_to_char_offsets, whitespace_word_spans

BACKBONE = "latincy/latin-bert"
SENTENCE = "Gallia est omnis divisa in partes tres"


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    backbone = argv[0] if argv else BACKBONE
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    # 1. tokenizer + word_ids -------------------------------------------------
    auto_tok: Any = AutoTokenizer
    tokenizer = auto_tok.from_pretrained(backbone, trust_remote_code=True, use_fast=True)
    words = [w for w, _, _ in whitespace_word_spans(SENTENCE)]
    enc = tokenizer(words, is_split_into_words=True, return_tensors="pt")
    word_ids = enc.word_ids(batch_index=0)
    covered = {w for w in word_ids if w is not None}
    check(
        "tokenizer.word_ids covers every word",
        covered == set(range(len(words))),
        f"{len(covered)}/{len(words)} words, ids={word_ids}",
    )

    # 2. model builds as token-classifier ------------------------------------
    auto_model: Any = AutoModelForTokenClassification
    model = auto_model.from_pretrained(
        backbone,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        trust_remote_code=True,
    )
    check("model.config.num_labels == 7", model.config.num_labels == NUM_LABELS)
    check(
        "model.config.id2label matches contract",
        {int(k): v for k, v in model.config.id2label.items()} == ID2LABEL,
        str(model.config.id2label),
    )

    # 3. forward pass --------------------------------------------------------
    model.eval()
    with torch.no_grad():
        logits = model(**enc).logits
    seq_len = enc["input_ids"].shape[1]
    check(
        "forward logits shape == [1, L, 7]",
        tuple(logits.shape) == (1, seq_len, NUM_LABELS),
        str(tuple(logits.shape)),
    )

    # 4. character-offset emission -------------------------------------------
    spans = whitespace_word_spans(SENTENCE)
    emitted = spans_to_char_offsets(SENTENCE, spans, [(0, 1, "GEO")])  # "Gallia" -> GEO
    ok_off = emitted == [{"text": "Gallia", "start": 0, "end": 6, "type": "GEO"}]
    check("character offsets emit correctly", ok_off, str(emitted))

    # bonus (non-fatal): the tokenizer's own offset_mapping is KNOWN to be
    # unreliable for LatinBERT (its escaping pre-tokenizer scrambles offsets),
    # which is exactly why offsets are derived from whitespace_word_spans above.
    try:
        raw = tokenizer(SENTENCE, return_offsets_mapping=True)
        first = next((s, e) for s, e in raw["offset_mapping"] if e > s)
        print(
            f"[INFO] tokenizer offset_mapping present (first content token {first}); "
            "NOT used for emission - whitespace_word_spans is the offset source of truth"
        )
    except Exception as exc:
        print(f"[INFO] tokenizer offset_mapping unavailable ({exc}); using whitespace_word_spans")

    if failures:
        print(f"\nGATE FAILED: {failures}")
        return 1
    print("\nGATE PASSED: LatinBERT is HF-loadable for token classification.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
