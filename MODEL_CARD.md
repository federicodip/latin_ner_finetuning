---
language: la
license: agpl-3.0
library_name: transformers
pipeline_tag: token-classification
tags:
  - latin
  - ner
  - classics
  - token-classification
base_model: latincy/latin-bert
---

# LatinBERT NER (classical Latin · PRS/GEO/GRP)

Fine-tuned [`latincy/latin-bert`](https://huggingface.co/latincy/latin-bert) (Bamman & Burns 2020
LatinBERT, BERT-base) for **named-entity recognition** on classical Latin, producing native
**PRS** (person), **GEO** (place), **GRP** (group) entities in IOB2.

- **Trained:** @@DATE@@
- **In-domain strict macro-F1:** @@MACRO_F1@@ (Herodotos prose held-out test)

## ⚠️ License — AGPL-3.0 (inherited from training data)

This checkpoint is a **derivative work** of the AGPL-3.0
[Herodotos Project Latin NER corpus](https://github.com/Herodotos-Project/Herodotos-Project-Latin-NER-Tagger-Annotation),
so **the weights are AGPL-3.0**. On redistribution **or exposure via a network service**, the
AGPL source-availability obligations apply. Internal use, training, and evaluation are unaffected.
The LatinBERT backbone itself is MIT/Apache-2.0; the copyleft comes from the *data*.

## Usage

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

model = AutoModelForTokenClassification.from_pretrained(CKPT_DIR, trust_remote_code=True)
tok   = AutoTokenizer.from_pretrained(CKPT_DIR, trust_remote_code=True)  # required: custom subword tokenizer
```

- **`trust_remote_code=True` is required** — LatinBERT uses a custom `LatinBertTokenizerFast`
  (faithful tensor2tensor `SubwordTextEncoder`). The tokenizer code is bundled in this dir, so it
  is self-contained (no internet needed).
- Labels (`id2label`): `0:O 1:B-PRS 2:I-PRS 3:B-GEO 4:I-GEO 5:B-GRP 6:I-GRP`.
- **Emits NATIVE PRS/GEO/GRP** — the downstream consumer maps `PRS→PER`, `GEO→LOC`, `GRP→NORP`.
- For NER, tokenize word-tokenized input with `is_split_into_words=True` and align predictions via
  `word_ids()`; character offsets are derived from whitespace word spans (see `latin_ner.offsets`).

## Training

- **Data:** Herodotos Project (Caesar *BG*+*BC*, Pliny *Elder*+*Younger*), pooled prose, fixed-seed
  stratified 75/12.5/12.5 sentence split. Ovid *Ars Amatoria* held out as poetry / out-of-domain.
- **Config (Beersmans et al. 2023, best in-domain):** lr 7.89e-5, weight_decay 0.10, 3 epochs,
  batch 16, warmup 0.1, lowercased input.
- **Expected:** in-domain macro-F1 ≈ 0.88–0.90; poetry (Ovid) ≈ 0.50 (large prose→poetry drop).

## Reproducibility

| Field | Value |
|---|---|
| backbone | `latincy/latin-bert` |
| repo git_sha | `@@GIT_SHA@@` |
| data git_sha | `@@DATA_SHA@@` |
| transformers | `@@TRANSFORMERS@@` |
| trained | `@@DATE@@` |

Full per-class strict + relaxed metrics: see `eval/latin_ner_eval.json` in the source repo.
