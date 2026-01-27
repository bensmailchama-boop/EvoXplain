#!/bin/bash
#SBATCH --job-name=bc_rf_agg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/bc_rf_agg_%j.log
#SBATCH --error=logs/bc_rf_agg_%j.err
#SBATCH --partition=core32
#SBATCH --chdir=/home2/chamabens/HPC_EvoXplain

# ============================================================================
# Aggregate BC RF Varied Experiment
# Step 1: Aggregate per-split chunks → .npz (with per-split clustering)
# Step 2: Universal aggregation + Disagreement report
# ============================================================================

set -euo pipefail

N_SPLITS=5
SPLIT_SEED_START=100
OUTPUT_DIR="results/bc_rf_shap_varied"

SCRIPT="evoxplain_core_engine.py"
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: $SCRIPT not found!"
    exit 1
fi

echo "=== BC-RF Varied - AGGREGATION ==="
echo "Output Dir: ${OUTPUT_DIR}"
echo "==================================="

# --- Environment check ---
echo ""
echo "=== Environment Check ==="
which python3.9
python3.9 --version
python3.9 -c "import shap; import sklearn; print('Packages OK')" || { echo "ERROR: Missing packages"; exit 1; }
echo "========================="

mkdir -p logs

# ----------------------------------------------------------------------------
# STEP 1: Aggregate each split (chunks → aggregate_split*.npz)
# ----------------------------------------------------------------------------
echo ""
echo ">>> STEP 1: Aggregating chunks for each split..."

for i in $(seq 0 $((N_SPLITS - 1))); do
    SPLIT_SEED=$((SPLIT_SEED_START + i))
    
    echo ""
    echo ">>> Aggregating Split ${i}/${N_SPLITS}: Seed=${SPLIT_SEED}"
    
    python3.9 ${SCRIPT} \
      --dataset breast_cancer \
      --model rf \
      --output_dir ${OUTPUT_DIR} \
      --mode aggregate_split \
      --split_seed ${SPLIT_SEED} \
      --rf_varied \
      --seed 42
    
    echo "✓ Split ${SPLIT_SEED} done"
done

# Check that npz files were created
echo ""
echo ">>> Checking aggregate files..."
ls -lh ${OUTPUT_DIR}/aggregate_split*.npz 2>/dev/null || { echo "ERROR: No aggregate_split*.npz files created!"; exit 1; }

# ----------------------------------------------------------------------------
# STEP 2: Universal Clustering & Disagreement
# ----------------------------------------------------------------------------
echo ""
echo ">>> STEP 2: Universal Clustering + Disagreement Analysis..."

SEEDS=$(seq -s, $SPLIT_SEED_START $((SPLIT_SEED_START + N_SPLITS - 1)))

python3.9 ${SCRIPT} \
  --dataset breast_cancer \
  --model rf \
  --output_dir ${OUTPUT_DIR} \
  --mode aggregate_universal \
  --split_seeds "${SEEDS}" \
  --disagreement \
  --rf_varied \
  --seed 42

echo ""
echo "========================================================================"
echo "✓ AGGREGATION COMPLETE!"
echo "========================================================================"
echo ""
echo "Files created:"
ls -lh ${OUTPUT_DIR}/aggregate_split*.npz 2>/dev/null
ls -lh ${OUTPUT_DIR}/*.csv 2>/dev/null || echo "  (no CSV files)"
ls -lh ${OUTPUT_DIR}/*.json 2>/dev/null || echo "  (no JSON files)"
echo ""
echo "Per-split summary:"
for i in $(seq 0 $((N_SPLITS - 1))); do
    SPLIT_SEED=$((SPLIT_SEED_START + i))
    echo -n "  Split ${SPLIT_SEED}: "
    python3.9 -c "
import numpy as np
d = np.load('${OUTPUT_DIR}/aggregate_split${SPLIT_SEED}.npz', allow_pickle=True)
k = d.get('best_k', [0])[0] if 'best_k' in d else 'N/A'
ent = d.get('entropy_norm', [0])[0] if 'entropy_norm' in d else 'N/A'
print(f'k={k}, entropy_norm={ent:.4f}' if isinstance(ent, float) else f'k={k}')
" 2>/dev/null || echo "(could not read)"
done
echo "========================================================================"
