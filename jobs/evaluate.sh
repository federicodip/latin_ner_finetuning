#!/usr/bin/bash -l
#SBATCH --job-name=latinner-eval
#SBATCH --time=00:30:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4
#SBATCH --partition=lowprio
#SBATCH --chdir=/home/fdipas/classical-latin-ner
#SBATCH --output=/home/fdipas/classical-latin-ner/logs/%x-%j.out
#SBATCH --error=/home/fdipas/classical-latin-ner/logs/%x-%j.err
set -e

# Standalone evaluation (CPU is enough for BERT-base inference over a few
# thousand sentences). Set CKPT to the checkpoint dir to evaluate.
REPO=/home/fdipas/classical-latin-ner
SCRATCH=/scratch/fdipas/classical-latin-ner
SIF=$SCRATCH/containers/finetune.sif
SPLITS=$SCRATCH/data/splits
CKPT=${CKPT:?set CKPT=/scratch/fdipas/classical-latin-ner/models/latin-bert-ner-<date>}
mkdir -p "$REPO/eval" "$REPO/logs"

GIT_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)
export HTTPS_PROXY=http://10.129.62.115:3128  # harmless if unused; covers any HF hub touch

if [ ! -f "$SIF" ]; then
  echo "ERROR: container not found: $SIF" >&2
  echo "Build it first:  sbatch jobs/build_container.sh" >&2
  exit 1
fi

module load apptainer

apptainer exec \
  --env PYTHONPATH="$REPO/src" \
  --env HTTPS_PROXY="$HTTPS_PROXY" \
  --env HF_HOME=/scratch/fdipas/cache/huggingface \
  "$SIF" python -m latin_ner.evaluate \
    --checkpoint "$CKPT" \
    --data-dir "$SPLITS" \
    --out-json "$REPO/eval/latin_ner_eval.json" \
    --out-md "$REPO/eval/latin_ner_eval.md" \
    --git-sha "$GIT_SHA"

cat "$REPO/eval/latin_ner_eval.md"
