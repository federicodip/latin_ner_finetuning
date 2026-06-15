#!/usr/bin/bash -l
#SBATCH --job-name=latinner-train
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
# SMALL job: BERT-base + ~5k entities = a few GPU-hours. NEVER request 80GB+.
# Constraint covers L4(24) / A100(80) / H100(80/96) / H200(140) and EXCLUDES
# V100 (GPUMEM32GB), which silently CPU-offloads on this cluster.
#SBATCH --constraint="GPUMEM24GB|GPUMEM80GB|GPUMEM96GB|GPUMEM140GB"
#SBATCH --partition=lowprio
#SBATCH --chdir=/home/fdipas/classical-latin-ner
#SBATCH --output=/home/fdipas/classical-latin-ner/logs/%x-%j.out
#SBATCH --error=/home/fdipas/classical-latin-ner/logs/%x-%j.err
set -e

REPO=/home/fdipas/classical-latin-ner
SCRATCH=/scratch/fdipas/classical-latin-ner
SIF=$SCRATCH/containers/finetune.sif
SPLITS=$SCRATCH/data/splits
DATE=$(date +%F)
OUTDIR=${OUTDIR:-$SCRATCH/models/latin-bert-ner-$DATE}
mkdir -p "$OUTDIR" "$REPO/logs" "$REPO/eval" /scratch/fdipas/cache/huggingface

export HTTPS_PROXY=http://10.129.62.115:3128
GIT_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)

# RESUME=1 continues from the latest checkpoint already in $OUTDIR.
RESUME_FLAG=""
if [ "${RESUME:-0}" = "1" ]; then RESUME_FLAG="--resume"; fi

if [ ! -f "$SIF" ]; then
  echo "ERROR: container not found: $SIF" >&2
  echo "Build it first:  sbatch jobs/build_container.sh" >&2
  exit 1
fi

module load apptainer

# ---- 1. fine-tune (GPU) -------------------------------------------------- #
apptainer exec --nv \
  --env PYTHONPATH="$REPO/src" \
  --env HTTPS_PROXY="$HTTPS_PROXY" \
  --env HF_HOME=/scratch/fdipas/cache/huggingface \
  "$SIF" python -m latin_ner.train \
    --data-dir "$SPLITS" \
    --output-dir "$OUTDIR" \
    --seed 13 \
    $RESUME_FLAG

# ---- 2. evaluate -> eval/*.json + .md (committed to the repo) ------------- #
apptainer exec --nv \
  --env PYTHONPATH="$REPO/src" \
  --env HTTPS_PROXY="$HTTPS_PROXY" \
  --env HF_HOME=/scratch/fdipas/cache/huggingface \
  "$SIF" python -m latin_ner.evaluate \
    --checkpoint "$OUTDIR" \
    --data-dir "$SPLITS" \
    --out-json "$REPO/eval/latin_ner_eval.json" \
    --out-md "$REPO/eval/latin_ner_eval.md" \
    --git-sha "$GIT_SHA"

# ---- 3. stamp + copy the model card into the checkpoint dir --------------- #
TFV=$(apptainer exec "$SIF" python -c "import transformers;print(transformers.__version__)")
DATA_SHA=$(apptainer exec "$SIF" python -c \
  "import json;print(json.load(open('$SPLITS/manifest.json')).get('data_git_sha','unknown'))")
MACRO=$(apptainer exec "$SIF" python -c \
  "import json;print(json.load(open('$REPO/eval/latin_ner_eval.json'))['acceptance']['in_domain_macro_f1'])")
sed -e "s|@@DATE@@|$DATE|g" \
    -e "s|@@GIT_SHA@@|$GIT_SHA|g" \
    -e "s|@@TRANSFORMERS@@|$TFV|g" \
    -e "s|@@DATA_SHA@@|$DATA_SHA|g" \
    -e "s|@@MACRO_F1@@|$MACRO|g" \
    "$REPO/MODEL_CARD.md" > "$OUTDIR/README.md"
echo "checkpoint + model card -> $OUTDIR"

# ---- 4. sanity guard (in-domain strict macro-F1) ------------------------- #
PASSED=$(apptainer exec "$SIF" python -c \
  "import json;print(json.load(open('$REPO/eval/latin_ner_eval.json'))['acceptance']['passes'])")
echo "in-domain strict macro-F1 = $MACRO (sanity threshold 0.85) -> passes=$PASSED"
if [ "$PASSED" != "True" ]; then
  echo "::SANITY WARNING:: in-domain macro-F1 below 0.85 — investigate before shipping." >&2
  exit 1
fi
echo "DONE."
