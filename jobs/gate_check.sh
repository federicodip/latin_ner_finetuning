#!/usr/bin/bash -l
#SBATCH --job-name=latinner-gate
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=lowprio
#SBATCH --chdir=/home/fdipas/classical-latin-ner
#SBATCH --output=/home/fdipas/classical-latin-ner/logs/%x-%j.out
#SBATCH --error=/home/fdipas/classical-latin-ner/logs/%x-%j.err
set -e

# FRICTION GATE — prove LatinBERT loads as AutoModelForTokenClassification and
# offsets work, BEFORE spending GPU time. CPU-only (model load + 1 forward pass).
REPO=/home/fdipas/classical-latin-ner
SCRATCH=/scratch/fdipas/classical-latin-ner
SIF=$SCRATCH/containers/finetune.sif
mkdir -p "$REPO/logs" /scratch/fdipas/cache/huggingface

export HTTPS_PROXY=http://10.129.62.115:3128

module load apptainer

# Non-zero exit (gate failure) propagates through set -e and fails the job.
apptainer exec \
  --env PYTHONPATH="$REPO/src" \
  --env HTTPS_PROXY="$HTTPS_PROXY" \
  --env HF_HOME=/scratch/fdipas/cache/huggingface \
  "$SIF" python -m latin_ner.gate_check
