#!/bin/bash
#SBATCH --job-name=compas_variedC_agg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/compas_variedC_agg_%j.log
#SBATCH --error=logs/compas_variedC_agg_%j.err

# =============================================================================
# USER CONFIGURATION - MODIFY THESE FOR YOUR ENVIRONMENT
# =============================================================================
WORKDIR="/path/to/your/EvoXplain"  # <-- CHANGE THIS

#SBATCH --partition=your_partition  # <-- CHANGE THIS

PYTHON="python3"  # <-- CHANGE IF NEEDED
# =============================================================================

cd "${WORKDIR}" || exit 1

set -euo pipefail

# =============================================================================
# CONFIGURATION - COMPAS Varied C Experiment
# =============================================================================
N_SPLITS=10
SPLIT_SEED_START=100
OUTPUT_DIR="results/compas_lr_shap_variedC"
SCRIPT="evoxplain_core_engine.py"

mkdir -p logs

echo "=== AGGREGATION JOB (COMPAS Varied C) ==="
echo "HOST=$(hostname)"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo ""

# =============================================================================
# STEP 1: Aggregate each split
# =============================================================================
echo "=== STEP 1: Aggregating individual splits ==="
for i in $(seq 0 $((N_SPLITS - 1))); do
    SPLIT_SEED=$((SPLIT_SEED_START + i))
    echo "Aggregating split ${SPLIT_SEED}..."
    
    ${PYTHON} ${SCRIPT} \
      --dataset compas \
      --model logreg \
      --mode aggregate_split \
      --split_seed ${SPLIT_SEED} \
      --output_dir ${OUTPUT_DIR}
done

echo ""
echo "=== STEP 2: Universal aggregation + clustering ==="

# Build comma-separated list of split seeds
SPLIT_SEEDS=""
for i in $(seq 0 $((N_SPLITS - 1))); do
    SPLIT_SEED=$((SPLIT_SEED_START + i))
    if [ -z "$SPLIT_SEEDS" ]; then
        SPLIT_SEEDS="${SPLIT_SEED}"
    else
        SPLIT_SEEDS="${SPLIT_SEEDS},${SPLIT_SEED}"
    fi
done

echo "Split seeds: ${SPLIT_SEEDS}"

${PYTHON} ${SCRIPT} \
  --dataset compas \
  --model logreg \
  --mode aggregate_universal \
  --split_seeds "${SPLIT_SEEDS}" \
  --output_dir ${OUTPUT_DIR} \
  --disagreement

echo ""
echo "=== FINAL OUTPUT FILES ==="
ls -lh ${OUTPUT_DIR}/universal_*.npy ${OUTPUT_DIR}/universal_*.json 2>/dev/null || echo "(no universal files found)"
ls -lh ${OUTPUT_DIR}/aggregate_split*.npz 2>/dev/null | head -5 || echo "(no aggregate files found)"
echo ""
echo "=== DONE ==="
