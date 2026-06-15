# Latin NER evaluation

- backbone: /scratch/fdipas/classical-latin-ner/models/latin-bert-ner-2026-06-15
- checkpoint: /scratch/fdipas/classical-latin-ner/models/latin-bert-ner-2026-06-15
- transformers_version: 5.8.1
- git_sha: 7466907372d2e8cb9d6597b3e5216365841e5e72
- data_git_sha: 229547aa0f64e6c13fdf6fe49f99383c0a243ea4
- num_labels: 7

## Acceptance
in_domain_test strict macro-F1 (floor; paper target 0.88-0.90) = 0.8373168814118319 (threshold 0.83) -> PASS

## Split: in_domain_test  (sentences=970, gold_entities=838, pred_entities=865)
STRICT  macro-F1=0.8373  micro P=0.8393 R=0.8663 F1=0.8526
  PRS  P=0.844 R=0.914 F1=0.878 support=467
  GEO  P=0.802 R=0.770 F1=0.786 support=174
  GRP  P=0.859 R=0.838 F1=0.848 support=197
RELAXED macro-F1=0.8691  micro P=0.8740 R=0.9021 F1=0.8878
  PRS  P=0.885 R=0.959 F1=0.921 support=467
  GEO  P=0.850 R=0.816 F1=0.833 support=174
  GRP  P=0.865 R=0.843 F1=0.853 support=197

## Split: poetry  (sentences=2435, gold_entities=571, pred_entities=537)
STRICT  macro-F1=0.4944  micro P=0.6034 R=0.5674 F1=0.5848
  PRS  P=0.676 R=0.658 F1=0.667 support=377
  GEO  P=0.473 R=0.299 F1=0.366 support=87
  GRP  P=0.435 R=0.467 F1=0.450 support=107
RELAXED macro-F1=0.5097  micro P=0.6350 R=0.5972 F1=0.6155
  PRS  P=0.722 R=0.703 F1=0.712 support=377
  GEO  P=0.473 R=0.299 F1=0.366 support=87
  GRP  P=0.435 R=0.467 F1=0.450 support=107

## Split: lasla_cross_genre  (sentences=505, gold_entities=1365, pred_entities=903)
STRICT  macro-F1=0.6154  micro P=0.8151 R=0.5363 F1=0.6469
  PRS  P=0.888 R=0.581 F1=0.702 support=859
  GEO  P=0.628 R=0.398 F1=0.487 support=314
  GRP  P=0.788 R=0.562 F1=0.657 support=192
RELAXED macro-F1=0.6688  micro P=0.8859 R=0.5861 F1=0.7055
  PRS  P=0.952 R=0.629 F1=0.757 support=859
  GEO  P=0.764 R=0.484 F1=0.593 support=314
  GRP  P=0.788 R=0.562 F1=0.657 support=192
