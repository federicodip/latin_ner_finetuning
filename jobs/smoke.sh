#!/usr/bin/bash -l
#SBATCH --job-name=latinner-smoke
#SBATCH --time=00:25:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="GPUMEM24GB|GPUMEM80GB|GPUMEM96GB|GPUMEM140GB"
#SBATCH --partition=lowprio
#SBATCH --chdir=/home/fdipas/classical-latin-ner
#SBATCH --output=/home/fdipas/classical-latin-ner/logs/%x-%j.out
#SBATCH --error=/home/fdipas/classical-latin-ner/logs/%x-%j.err
set -e

# SMOKE: validate the full train loop + that the saved checkpoint reloads
# self-contained and OFFLINE (the OUTPUT CONTRACT) BEFORE the real finetune.
# Trains 1 epoch on 64 sentences (seconds of compute) to a throwaway dir.
REPO=/home/fdipas/classical-latin-ner
SCRATCH=/scratch/fdipas/classical-latin-ner
SIF=$SCRATCH/containers/finetune.sif
SPLITS=$SCRATCH/data/splits
SMOKE_OUT=$SCRATCH/models/_smoke
rm -rf "$SMOKE_OUT"; mkdir -p "$SMOKE_OUT" "$REPO/logs" /scratch/fdipas/cache/huggingface

export HTTPS_PROXY=http://10.129.62.115:3128

if [ ! -f "$SIF" ]; then
  echo "ERROR: container not found: $SIF" >&2
  echo "Build it first:  sbatch jobs/build_container.sh" >&2
  exit 1
fi

module load apptainer

# ---- 1. tiny training run (downloads backbone on first use) -------------- #
apptainer exec --nv \
  --env PYTHONPATH="$REPO/src" \
  --env HTTPS_PROXY="$HTTPS_PROXY" \
  --env HF_HOME=/scratch/fdipas/cache/huggingface \
  "$SIF" python -m latin_ner.train \
    --data-dir "$SPLITS" \
    --output-dir "$SMOKE_OUT" \
    --epochs 1 \
    --max-train-samples 64

# ---- 2. prove the checkpoint reloads OFFLINE + self-contained ------------ #
# HF_HUB_OFFLINE=1 forces loading from the saved dir only (no internet). If the
# custom tokenizer code wasn't bundled, or the head isn't a 7-label classifier,
# the gate FAILS here and exits non-zero (set -e).
apptainer exec \
  --env PYTHONPATH="$REPO/src" \
  --env HF_HOME=/scratch/fdipas/cache/huggingface \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  "$SIF" python -m latin_ner.gate_check "$SMOKE_OUT"

echo "SMOKE OK: train loop runs and the checkpoint reloads self-contained + offline."
