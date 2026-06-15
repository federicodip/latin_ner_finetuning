"""Fine-tune LatinBERT for token-classification NER with the HF Trainer.

Hyperparameters default to Beersmans et al. 2023's best *in-domain* config
(macro-F1 ~0.90): lr 7.89e-5, weight_decay 0.10, 3 epochs, batch 16, warmup 0.1,
lowercased input (the LatinBERT tokenizer lowercases by default).

The pure pieces (config, eval-prediction decoding/metrics) are unit-tested
offline. ``train`` / ``main`` lazily import torch + transformers and run on the
cluster, checkpointing to /scratch with ``--resume`` support.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .align import align_labels_to_subwords
from .evaluate import strict_scores
from .labels import ID2LABEL, LABEL2ID, NUM_LABELS, encode


@dataclass(frozen=True)
class TrainConfig:
    """Fine-tuning configuration (defaults = Beersmans best in-domain)."""

    backbone: str = "latincy/latin-bert"
    data_dir: str = ""
    output_dir: str = ""
    learning_rate: float = 7.89e-5
    weight_decay: float = 0.10
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    warmup_ratio: float = 0.1
    max_length: int = 256
    seed: int = 13
    label_all_subwords: bool = False
    resume: bool = False
    max_train_samples: int | None = None  # None = full corpus; int = fast smoke subset


# --------------------------------------------------------------------------- #
# Pure: eval-prediction decoding + metrics
# --------------------------------------------------------------------------- #
def decode_eval(
    pred_ids: Sequence[Sequence[int]],
    label_ids: Sequence[Sequence[int]],
) -> tuple[list[list[str]], list[list[str]]]:
    """Turn per-token prediction/label id matrices into BIO label sequences,
    dropping positions where the gold label is the ignore index (-100)."""
    gold: list[list[str]] = []
    pred: list[list[str]] = []
    for p_row, l_row in zip(pred_ids, label_ids, strict=True):
        g_seq: list[str] = []
        p_seq: list[str] = []
        for p, lab in zip(p_row, l_row, strict=True):
            if int(lab) == -100:
                continue
            g_seq.append(ID2LABEL[int(lab)])
            p_seq.append(ID2LABEL[int(p)])
        gold.append(g_seq)
        pred.append(p_seq)
    return gold, pred


def compute_metrics_from_arrays(
    pred_ids: Sequence[Sequence[int]],
    label_ids: Sequence[Sequence[int]],
) -> dict[str, float]:
    """Strict entity-level metrics for the Trainer's eval loop."""
    gold, pred = decode_eval(pred_ids, label_ids)
    s = strict_scores(gold, pred)
    return {
        "macro_f1": s["macro_f1"],
        "micro_f1": s["micro"]["f1"],
        "precision": s["micro"]["precision"],
        "recall": s["micro"]["recall"],
    }


def build_compute_metrics() -> Callable[[Any], dict[str, float]]:
    """Return a ``compute_metrics`` callable for the HF Trainer.

    Pairs with :func:`preprocess_logits_for_metrics`, which argmaxes logits to
    ids first (so ``eval_pred.predictions`` is already a [N, L] id matrix)."""

    def compute(eval_pred: Any) -> dict[str, float]:
        preds, labels = eval_pred
        return compute_metrics_from_arrays(preds, labels)

    return compute


# --------------------------------------------------------------------------- #
# Cluster glue (lazy torch/transformers import; runs on GPU node)
# --------------------------------------------------------------------------- #
def tokenize_and_align(  # pragma: no cover
    batch: Any,
    tokenizer: Any,
    *,
    label_all_subwords: bool,
    max_length: int,
) -> Any:
    enc = tokenizer(
        batch["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )
    aligned: list[list[int]] = []
    for i, labs in enumerate(batch["labels"]):
        word_ids = enc.word_ids(batch_index=i)
        aligned.append(
            align_labels_to_subwords(word_ids, encode(labs), label_all_subwords=label_all_subwords)
        )
    enc["labels"] = aligned
    return enc


def preprocess_logits_for_metrics(logits: Any, labels: Any) -> Any:  # pragma: no cover
    return logits.argmax(dim=-1)


def train(config: TrainConfig) -> str:  # pragma: no cover
    """Fine-tune and save a checkpoint dir; returns the output path."""
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(config.seed)

    auto_tok: Any = AutoTokenizer
    auto_model: Any = AutoModelForTokenClassification
    # Fast tokenizer: needed for word_ids() during label alignment. Its custom
    # Python pre-tokenizer can't be serialized to tokenizer.json, and on
    # transformers v5 use_fast=False does NOT give a serializable slow tokenizer
    # either -> we must never hand a tokenizer to the Trainer to save (see the
    # collator + processing_class note below).
    tokenizer = auto_tok.from_pretrained(config.backbone, trust_remote_code=True, use_fast=True)
    model = auto_model.from_pretrained(
        config.backbone,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        trust_remote_code=True,
    )

    data_files = {
        "train": f"{config.data_dir}/train.jsonl",
        "validation": f"{config.data_dir}/dev.jsonl",
    }
    ds = load_dataset("json", data_files=data_files)
    if config.max_train_samples is not None:
        n = config.max_train_samples
        ds["train"] = ds["train"].select(range(min(n, len(ds["train"]))))
        ds["validation"] = ds["validation"].select(range(min(n, len(ds["validation"]))))
    ds = ds.map(
        lambda b: tokenize_and_align(
            b,
            tokenizer,
            label_all_subwords=config.label_all_subwords,
            max_length=config.max_length,
        ),
        batched=True,
        remove_columns=["tokens", "labels"],
    )

    args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        warmup_ratio=config.warmup_ratio,
        eval_strategy="epoch",  # transformers v5 name (was evaluation_strategy)
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        seed=config.seed,
        report_to="none",
    )

    # CRITICAL: never hand the Trainer a tokenizer. The Trainer saves its
    # processing_class at EVERY checkpoint, and any LatinBERT tokenizer object
    # (fast, or v5 "slow" which is fast-backed) raises "Custom PreTokenizer
    # cannot be serialized". So we use a plain-function collator (no .tokenizer
    # attribute for the Trainer to adopt) and omit processing_class -> the
    # Trainer never calls save_pretrained. We copy the tokenizer source files
    # into the checkpoint ourselves via _save_tokenizer_sources.
    pad_id = tokenizer.pad_token_id

    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        width = max(len(f["input_ids"]) for f in features)
        input_ids, attention, labels = [], [], []
        for f in features:
            gap = width - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * gap)
            attention.append(f["attention_mask"] + [0] * gap)
            labels.append(f["labels"] + [-100] * gap)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collate,
        compute_metrics=build_compute_metrics(),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train(resume_from_checkpoint=config.resume or None)
    trainer.save_model(config.output_dir)
    _save_tokenizer_sources(config.backbone, config.output_dir)
    return config.output_dir


# Tokenizer files the Hub backbone ships; copied verbatim so the checkpoint can
# rebuild the fast tokenizer (use_fast=True) offline. tokenizer.json is absent
# by design (the custom pre-tokenizer is rebuilt from latin.subword.encoder).
# REQUIRED files MUST be present or the checkpoint can't load use_fast offline;
# the rest are best-effort.
_TOKENIZER_FILES_REQUIRED = (
    "latin.subword.encoder",
    "tokenization_latin_bert.py",
    "tokenization_latin_bert_fast.py",
    "tokenizer_config.json",
)
_TOKENIZER_FILES_OPTIONAL = ("special_tokens_map.json",)


def _save_tokenizer_sources(backbone: str, out_dir: str) -> None:  # pragma: no cover
    """Copy the backbone's tokenizer source files into the checkpoint so it is
    self-contained for `AutoTokenizer.from_pretrained(dir, use_fast=True,
    trust_remote_code=True)`.

    Raises RuntimeError if a REQUIRED file can't be fetched — failing loudly at
    save time beats shipping a checkpoint that silently can't reload offline.
    """
    import shutil
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    out = Path(out_dir)
    for fname in _TOKENIZER_FILES_REQUIRED:
        try:
            src = hf_hub_download(backbone, fname)
        except Exception as exc:
            raise RuntimeError(
                f"could not fetch required tokenizer file {fname!r} from {backbone!r}; "
                f"the checkpoint would not reload with use_fast=True offline"
            ) from exc
        shutil.copy(src, out / fname)
    for fname in _TOKENIZER_FILES_OPTIONAL:
        try:
            src = hf_hub_download(backbone, fname)
        except Exception:
            continue  # genuinely optional (some backbones omit it)
        shutil.copy(src, out / fname)


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Fine-tune LatinBERT for Latin NER.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone", default="latincy/latin-bert")
    parser.add_argument("--learning-rate", type=float, default=7.89e-5)
    parser.add_argument("--weight-decay", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-train-samples", type=int, default=None, help="subset train+val for a fast smoke run"
    )
    args = parser.parse_args(argv)

    config = TrainConfig(
        backbone=args.backbone,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        max_length=args.max_length,
        seed=args.seed,
        resume=args.resume,
        max_train_samples=args.max_train_samples,
    )
    out = train(config)
    print(f"saved checkpoint -> {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
