#!/usr/bin/bash -l
#SBATCH --job-name=latinner-data
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=lowprio
#SBATCH --chdir=/home/fdipas/classical-latin-ner
#SBATCH --output=/home/fdipas/classical-latin-ner/logs/%x-%j.out
#SBATCH --error=/home/fdipas/classical-latin-ner/logs/%x-%j.err
set -e

# Clone the AGPL Herodotos corpus (+ optional CC-BY-NC-SA LASLA eval data) onto
# /scratch and build the fixed-seed stratified splits. CPU-only.
REPO=/home/fdipas/classical-latin-ner
SCRATCH=/scratch/fdipas/classical-latin-ner
RAW=$SCRATCH/data/raw
SPLITS=$SCRATCH/data/splits
SIF=$SCRATCH/containers/finetune.sif
mkdir -p "$RAW" "$SPLITS" "$REPO/logs"

export HTTPS_PROXY=http://10.129.62.115:3128
export HTTP_PROXY=http://10.129.62.115:3128

# Herodotos Project Latin NER (AGPL-3.0) — the training corpus.
HERODOTOS="$RAW/Herodotos-Project-Latin-NER-Tagger-Annotation"
if [ ! -d "$HERODOTOS" ]; then
  git clone --depth 1 \
    https://github.com/Herodotos-Project/Herodotos-Project-Latin-NER-Tagger-Annotation.git \
    "$HERODOTOS"
fi

# Ner-Latin-RANLP (CC-BY-NC-SA gold data) — optional LASLA cross-genre eval.
RANLP="$RAW/Ner-Latin-RANLP"
if [ ! -d "$RANLP" ]; then
  git clone --depth 1 https://github.com/NER-AncientLanguages/Ner-Latin-RANLP.git "$RANLP"
fi

if [ ! -f "$SIF" ]; then
  echo "ERROR: container not found: $SIF" >&2
  echo "Build it first:  sbatch jobs/build_container.sh  (or chain with --dependency=afterok)" >&2
  exit 1
fi

module load apptainer

# No GPU needed (--nv omitted). PYTHONPATH points at the src-layout package.
apptainer exec \
  --env PYTHONPATH="$REPO/src" \
  --env HTTPS_PROXY="$HTTPS_PROXY" \
  "$SIF" python -m latin_ner.data \
    --herodotos-dir "$HERODOTOS/Annotation_1-1-19" \
    --lasla-dir "$RANLP/Latin_Gold_Data" \
    --out-dir "$SPLITS" \
    --seed 13

echo "==== manifest ===="
cat "$SPLITS/manifest.json"
