#!/usr/bin/bash -l
#SBATCH --job-name=latinner-build
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=lowprio
#SBATCH --chdir=/home/fdipas/classical-latin-ner
#SBATCH --output=/home/fdipas/classical-latin-ner/logs/%x-%j.out
#SBATCH --error=/home/fdipas/classical-latin-ner/logs/%x-%j.err
set -e

# Build the Apptainer container for fine-tuning. CPU-only job (no GPU needed).
REPO=/home/fdipas/classical-latin-ner
SCRATCH=/scratch/fdipas/classical-latin-ner
mkdir -p "$SCRATCH/containers" "$REPO/logs"

module load apptainer

# APPTAINER_BINDPATH="" — cluster auto-binds break builds.
# --ignore-fakeroot-command — host glibc older than the base image.
HTTPS_PROXY=http://10.129.62.115:3128 HTTP_PROXY=http://10.129.62.115:3128 \
    APPTAINER_BINDPATH="" \
    apptainer build --ignore-fakeroot-command \
    "$SCRATCH/containers/finetune.sif" \
    "$REPO/containers/finetune.def"

echo "built: $SCRATCH/containers/finetune.sif"
