#!/bin/bash
#SBATCH --job-name=adult_aggregate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/adult_aggregate_%j.log
#SBATCH --error=logs/adult_aggregate_%j.log
#SBATCH --partition=core32
#SBATCH --chdir=/home2/chamabens/HPC_EvoXplain

set -euo pipefail

# =============================================================================
# CONFIGURATION (Adult Dataset - Varied C)
# =============================================================================
N_SPLITS=10
SPLIT_SEED_START=100
OUTPUT_DIR="results/adult_lr_shap_variedC"
SCRIPT="evoxplain_core_engine.py"

mkdir -p logs

echo "=== AGGREGATION JOB (Adult Dataset, Varied C) ==="
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
    
    python3.9 ${SCRIPT} \
      --dataset adult \
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

python3.9 ${SCRIPT} \
  --dataset adult \
  --model logreg \
  --mode aggregate_universal \
  --split_seeds "${SPLIT_SEEDS}" \
  --output_dir ${OUTPUT_DIR} \
  --disagreement

echo ""
echo "=== FINAL OUTPUT FILES ==="
ls -lh ${OUTPUT_DIR}/universal_*.npy ${OUTPUT_DIR}/universal_*.json 2>/dev/null || echo "(no universal files found)"
echo ""
echo "=== DONE ==="
