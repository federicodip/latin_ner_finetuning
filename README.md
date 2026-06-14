# classical-latin-ner

Fine-tune **LatinBERT** into a reusable, loadable **classical-Latin NER** checkpoint that emits
native **PRS / GEO / GRP** entities with character offsets.

This repo's *sole* purpose is to produce that checkpoint. It is **standalone on purpose** — it
exists separately from the main AntiquityGPT project to **quarantine the AGPL-3.0 training data**
(Herodotos Project) away from that MIT codebase. Nothing here depends on AntiquityGPT; training and
evaluation use only public Latin NER data.

---

## ⚠️ License & derivative status — read this first

| Component | License | Implication |
|---|---|---|
| **This repo's code** | **AGPL-3.0-or-later** (`LICENSE`) | Copyleft. |
| **Training data** — [Herodotos Project Latin NER](https://github.com/Herodotos-Project/Herodotos-Project-Latin-NER-Tagger-Annotation) | **AGPL-3.0** | The data is copyleft. |
| **Resulting fine-tuned checkpoint** | **AGPL-3.0 (inherited)** | A model trained on AGPL data is a **derivative work**. On redistribution **or network-service exposure**, the AGPL source-availability obligations apply. |
| **LatinBERT backbone** — [`latincy/latin-bert`](https://huggingface.co/latincy/latin-bert) | MIT weights, Apache-2.0 packaging | The backbone itself is permissive; the AGPL obligation comes from the *data*, not the backbone. |
| **Optional LASLA cross-genre eval data** ([Ner-Latin-RANLP](https://github.com/NER-AncientLanguages/Ner-Latin-RANLP)) | **CC BY-NC-SA 4.0** (non-commercial) | Used for *evaluation only*, kept as a separate split, never blended into training. |

**Internal training + evaluation is unaffected.** The AGPL obligations bite only on *redistribution*
or *exposing the model over a network service*. This is restated in the checkpoint's model card.

---

## OUTPUT CONTRACT (how the downstream adapter consumes this — keep stable)

The deliverable is a **standard HuggingFace token-classification checkpoint**:

1. Loadable with:
   ```python
   from transformers import AutoModelForTokenClassification, AutoTokenizer
   model = AutoModelForTokenClassification.from_pretrained(CKPT_DIR, trust_remote_code=True)
   tok   = AutoTokenizer.from_pretrained(CKPT_DIR, trust_remote_code=True)
   ```
   > `trust_remote_code=True` is **required** — LatinBERT uses a custom subword tokenizer
   > (`LatinBertTokenizerFast`, a faithful reimplementation of the tensor2tensor `SubwordTextEncoder`).
   > The tokenizer code travels *inside* the checkpoint dir, so it is self-contained (no internet).
2. `config.json` carries a correct **`id2label` / `label2id`** over the 7-label BIO set:
   ```
   0:O  1:B-PRS  2:I-PRS  3:B-GEO  4:I-GEO  5:B-GRP  6:I-GRP
   ```
3. Emits **NATIVE PRS / GEO / GRP** with **character offsets**. The consumer maps
   `PRS→PER`, `GEO→LOC`, `GRP→NORP` itself — **this repo does NOT remap.**
4. Checkpoint location (gitignored, lives on scratch):
   ```
   /scratch/fdipas/classical-latin-ner/models/latin-bert-ner-<YYYY-MM-DD>/
   ```

---

## Data

**Primary (Herodotos Project, AGPL-3.0)** — BIO-tagged PRS/GEO/GRP over:

| Work | File | Domain | Role |
|---|---|---|---|
| Caesar, *Bellum Gallicum* | `GWtrain.crf` + `GWtest.crf` | prose | in-domain |
| Caesar, *Bellum Civile* | `CW.crf` | prose | in-domain |
| Pliny the Elder, *Naturalis Historia* | `PlinyElder.crf` | prose | in-domain |
| Pliny the Younger, *Epistulae* | `PlinyYounger.crf` | prose | in-domain |
| Ovid, *Ars Amatoria* | `Ovid.crf` | **poetry** | **out-of-domain test only** |

- **Format gotcha:** the gold files are **CRFsuite** (`.crf`), *not* CoNLL. Layout is **label-first**:
  `LABEL\tTOKEN`, blank line between sentences, outside label is the digit `0`, and entity tags use
  **suffix** notation (`PRS-B`, `PRS-I`). `src/latin_ner/conll.py` normalises these to canonical
  IOB2 (`B-PRS`, `I-PRS`, `O`). Scheme is **IOB2** → seqeval `scheme="IOB2"`.
- **Split:** the repo ships no canonical multi-author split (only a Gallic-War train/test). Following
  Beersmans et al. 2023, we **pool the 4 prose works** and take a **fixed-seed stratified 75 / 12.5 /
  12.5** sentence-level split. Ovid is held out entirely as the poetry cross-genre test.
- Corpus ≈ **135,875 tokens / 7,175 entity mentions** (PRS 4,872 · GRP 1,841 · GEO 1,362).

**Optional secondary (LASLA / Ner-Latin-RANLP, CC BY-NC-SA 4.0)** — Tacitus *Historiae*, Cicero
*Philippicae*, Juvenal *Saturae* (poetry). 5-column TSV with UD-style multiword rows. Used as a
**separate cross-genre eval split** with a documented remap (`PER→PRS`, `DATE` dropped). Never blended
into training.

Raw data lives on `/scratch`, **never in git** (see `.gitignore`).

---

## Backbone — why `latincy/latin-bert`

Web-verified mid-2026. `latincy/latin-bert` is the genuine Bamman & Burns (2020) LatinBERT
(`BertModel`, 12 layers, 768 hidden, vocab 32,900, ~111M params) repackaged for HF, **with a
Rust-backed fast tokenizer exposing `word_ids()`** — exactly what NER subword→word label alignment
needs. (Rejected `pnadel/LatinBERT`: it is a 6-layer RoBERTa trained from scratch with *no* tokenizer
— not the real LatinBERT.)

---

## Cluster workflow (UZH ScienceCluster, user `fdipas`)

```bash
# 0. local: push; cluster: pull + fix Windows line endings
git pull && sed -i 's/\r$//' jobs/*.sh

# 1. build the Apptainer container (one-off)
sbatch jobs/build_container.sh

# 2. download + parse + split the Herodotos data onto /scratch
sbatch jobs/prepare_data.sh

# 3. FRICTION GATE — prove AutoModelForTokenClassification builds + offsets are correct
sbatch jobs/gate_check.sh        # must pass before training

# 4. fine-tune (checkpoints to /scratch, resumable)
sbatch jobs/finetune.sh          # add RESUME=1 to continue from last checkpoint

# 5. evaluate → eval/latin_ner_eval.json + eval/latin_ner_eval.md
sbatch jobs/evaluate.sh
```

**Acceptance gate:** in-domain entity-level **macro-F1 ≈ 0.88–0.90** (matches Beersmans 2023);
poetry split expected to drop to **≈ 0.50** (reported honestly, not gated on).

---

## Repo layout

```
README.md  LICENSE  pyproject.toml  requirements.txt
src/latin_ner/
  labels.py     # BIO label <-> id  (pure, unit-tested)
  conll.py      # .crf / CoNLL parse + IOB2 repair  (pure, unit-tested)
  spans.py      # BIO->spans + strict/relaxed P/R/F1  (pure, unit-tested)
  align.py      # word_ids subword<->word label alignment  (pure, unit-tested)
  data.py       # download + stratified split (writes /scratch)
  train.py      # HF Trainer fine-tune (--resume, scratch checkpoints)
  evaluate.py   # seqeval strict + relaxed, per-class; emits eval/*
containers/finetune.def
jobs/           # Slurm: build_container, prepare_data, gate_check, finetune, evaluate
tests/          # pytest (TDD, 80%+ coverage on pure helpers)
eval/           # latin_ner_eval.json + .md  (tracked)
data/           # gitignored; Herodotos clone lives on /scratch
```

## Reproducibility

Every eval report + the model card stamps: backbone id, data git-sha, hyperparameters,
`transformers` version, and this repo's git-sha.
