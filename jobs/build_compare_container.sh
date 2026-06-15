#!/usr/bin/bash -l
#SBATCH --job-name=latinner-cmpbuild
#SBATCH --time=01:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=lowprio
#SBATCH --chdir=/home/fdipas/classical-latin-ner
#SBATCH --output=/home/fdipas/classical-latin-ner/logs/%x-%j.out
#SBATCH --error=/home/fdipas/classical-latin-ner/logs/%x-%j.err
set -e

# Build the conda-based LatinCy comparison container (spaCy + la_core_web_lg).
# CPU-only. Conda + pip + the HF wheel all need internet via the proxy.
REPO=/home/fdipas/classical-latin-ner
SCRATCH=/scratch/fdipas/classical-latin-ner
mkdir -p "$SCRATCH/containers" "$REPO/logs"

module load apptainer

# Proxy for conda-forge + PyPI + HF. APPTAINER_BINDPATH="" + --ignore-fakeroot-command
# per the cluster's no-fakeroot build constraint (same as the training container).
HTTPS_PROXY=http://10.129.62.115:3128 HTTP_PROXY=http://10.129.62.115:3128 \
    APPTAINER_BINDPATH="" \
    apptainer build --ignore-fakeroot-command \
    "$SCRATCH/containers/compare.sif" \
    "$REPO/containers/compare.def"

echo "built: $SCRATCH/containers/compare.sif"
