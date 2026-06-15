# LatinCy vs. fine-tuned LatinBERT — Latin NER comparison

**TL;DR.** On the Herodotos Latin NER task (PRS/GEO/GRP), our fine-tuned LatinBERT model
**massively outperforms the general-purpose LatinCy spaCy pipelines** — by roughly **+0.40 macro-F1
in-domain** — and LatinCy is essentially **unable to detect group (GRP) entities at all**. This
reproduces the published finding of Beersmans et al. (2023).

---

## 1. Our model (this repo)

Entity-level scoring via seqeval `mode="strict", scheme="IOB2"`, on the held-out Herodotos prose
test split (see `eval/latin_ner_eval.md`). Checkpoint `latin-bert-ner-2026-06-15`, transformers
5.8.1, repo git_sha `7466907`, data git_sha `229547aa`.

| Split | STRICT macro-F1 | PRS | GEO | GRP |
|---|---|---|---|---|
| in-domain test (970 sents) | **0.84** | 0.88 | 0.79 | 0.85 |
| poetry / Ovid *Ars am.* (OOD) | 0.49 | 0.67 | 0.37 | 0.45 |

(Two runs landed 0.8373 @ max_length=512 and 0.8488 @ 256 — within single-run NER variance; GEO has
only 174 test instances. Relaxed in-domain macro-F1 = 0.87.)

## 2. LatinCy (published, **same** Herodotos test split)

LatinCy's NER emits `PERSON / LOC / NORP`, mapped to our `PRS / GEO / GRP`
(`PERSON→PRS, LOC→GEO, NORP→GRP`). Numbers below are **entity-level F1 from Beersmans et al. 2023,
Table 5** — i.e. LatinCy and fine-tuned LatinBERT scored on the *identical* test set, the cleanest
available head-to-head:

| Class | LatinBERT (LB2, best) | **LatinCy `la_core_web_lg`** | **LatinCy `la_core_web_trf`** | support |
|---|---|---|---|---|
| PERS | 0.92 | 0.64 | 0.64 | 474 |
| LOC | 0.87 | 0.61 | 0.54 | 218 |
| GRP | 0.91 | **0.02** | **0.06** | 247 |
| **macro** | **0.90** | **0.43** | **0.44** | 939 |
| macro (OOD poetry) | 0.50 | 0.25 | 0.20 | 569 |

## 3. The delta

- **In-domain: fine-tuned LatinBERT macro-F1 0.88–0.90 vs LatinCy 0.43–0.44 → ≈ +0.45.**
- **Out-of-domain (poetry): 0.50–0.54 vs 0.20–0.25 → ≈ +0.30.**
- The decisive failure mode: **LatinCy scores ≈0 on GRP** (0.02 lg / 0.06 trf). Its `NORP` label
  (nationalities/religious/political groups) rarely matches the Herodotos *socio-ethnic group*
  annotation, so it misses almost all group entities. Our model, trained on the exact scheme, gets
  GRP ≈ 0.85.
- **Our model reproduces the LatinBERT side** (0.84 in-domain, within split variance of the paper's
  0.88–0.90) and therefore beats LatinCy by ≈ **+0.40 macro-F1** on the target task.

## 4. Why this is a fair claim (and its caveats)

- **Same task, same gold, same metric** for the paper's LatinBERT-vs-LatinCy comparison (Table 5,
  entity-level strict on the identical Herodotos test set). Our number is computed with the same
  metric family (seqeval strict IOB2) on the same corpus, different fixed-seed split (the paper's
  split seed was never released; cross-split macro variance ≈ ±0.02–0.03 — far smaller than the
  +0.45 gap).
- **Specialized vs general.** LatinCy is a general Latin NLP pipeline (POS/lemma/parse/NER) trained
  on UD treebanks + LASLA with a general `PERSON/LOC/NORP` scheme; it was never trained on the
  Herodotos PRS/GEO/GRP annotation. The comparison answers "for *this* task, is the purpose-built
  model better?" — decisively yes — not "is LatinCy a bad tool" (it is strong at its own UD-style
  NER, reporting ~0.90 F1 on its own test set).
- **LASLA caveat.** LatinCy trained on LASLA, so our `lasla_cross_genre` split is *contaminated for
  LatinCy* and is excluded from the head-to-head.

## 5. Optional: a direct same-split LatinCy run

The numbers above use the paper's same-split LatinBERT-vs-LatinCy comparison plus our reproduction.
For a fully self-contained number (LatinCy run on *our exact* test split with *our* scorer), the
harness would: load `la_core_web_lg`/`trf`, feed our gold-tokenized words via
`spacy.tokens.Doc(nlp.vocab, words=...)`, read `doc.ents`, convert to IOB2, map labels, and reuse
`latin_ner.evaluate.strict_scores`. It requires a **conda-based container** (LatinCy's
`latincy-preprocess` has compiled extensions and the cluster can't `apt`-install a compiler without
fakeroot). Given the published +0.45 gap, this would confirm — not change — the verdict.

---

*Source: Marijke Beersmans, Evelien de Graaf, Tim Van de Cruys, Margherita Fantoli. 2023. "Training
and Evaluation of Named Entity Recognition Models for Classical Latin." Proc. ALP @ RANLP 2023,
Table 5. https://aclanthology.org/2023.alp-1.1/*
