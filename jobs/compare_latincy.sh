#!/usr/bin/bash -l
#SBATCH --job-name=latinner-compare
#SBATCH --time=00:30:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4
#SBATCH --partition=lowprio
#SBATCH --chdir=/home/fdipas/classical-latin-ner
#SBATCH --output=/home/fdipas/classical-latin-ner/logs/%x-%j.out
#SBATCH --error=/home/fdipas/classical-latin-ner/logs/%x-%j.err
set -e

# Run LatinCy (la_core_web_lg) on OUR exact test/poetry splits, score with OUR
# seqeval scorer, and tabulate the delta vs our fine-tuned model. CPU-only,
# fully offline (the model is bundled in the container; our scores come from
# the committed eval/latin_ner_eval.json).
REPO=/home/fdipas/classical-latin-ner
SCRATCH=/scratch/fdipas/classical-latin-ner
SIF=$SCRATCH/containers/compare.sif
SPLITS=$SCRATCH/data/splits
mkdir -p "$REPO/eval" "$REPO/logs"

if [ ! -f "$SIF" ]; then
  echo "ERROR: container not found: $SIF" >&2
  echo "Build it first:  sbatch jobs/build_compare_container.sh" >&2
  exit 1
fi
if [ ! -f "$REPO/eval/latin_ner_eval.json" ]; then
  echo "ERROR: $REPO/eval/latin_ner_eval.json not found (our model's scores)." >&2
  echo "Run jobs/finetune.sh first and commit eval/, or git pull it on the cluster." >&2
  exit 1
fi

module load apptainer

apptainer exec \
  --env PYTHONPATH="$REPO/src" \
  "$SIF" python -m latin_ner.compare \
    --model la_core_web_lg \
    --data-dir "$SPLITS" \
    --our-eval "$REPO/eval/latin_ner_eval.json" \
    --out-json "$REPO/eval/latincy_comparison_direct.json" \
    --out-md "$REPO/eval/latincy_comparison_direct.md"

echo "==== comparison ===="
cat "$REPO/eval/latincy_comparison_direct.md"
