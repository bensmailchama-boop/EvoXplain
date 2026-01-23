#!/bin/bash
#SBATCH --job-name=adult_rf_varied_agg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/adult_rf_varied_agg_%j.log
#SBATCH --error=logs/adult_rf_varied_agg_%j.err
#SBATCH --partition=core32
#SBATCH --chdir=/home2/chamabens/HPC_EvoXplain

# ============================================================================
# Aggregate RF Varied Experiment (Adult Income)
# Step 1: Aggregate per-split chunks -> .npz
# Step 2: Universal clustering + Disagreement report
# ============================================================================

set -euo pipefail

# --- Configuration ---
N_SPLITS=5
SPLIT_SEED_START=100
OUTPUT_DIR="results/adult_rf_shap_varied"

# --- Point to the Core Engine Script ---
SCRIPT="evoxplain_core_engine.py"
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: $SCRIPT not found. Please ensure the python script is present."
    exit 1
fi

# --- Verify Adult Dataset Exists ---
ADULT_DATA="data/adult.csv"
if [ ! -f "$ADULT_DATA" ]; then
    echo "ERROR: Adult dataset not found at ${ADULT_DATA}"
    echo "Aggregation requires dataset for disagreement analysis."
    echo "Please run: bash download_adult_data.sh"
    exit 1
fi

echo "=== Environment Check ==="
echo "Host: $(hostname)"
echo "CWD: $(pwd)"
echo "Script: $SCRIPT"
echo "Adult Data: $ADULT_DATA"
echo "Output Dir: $OUTPUT_DIR"
echo "Python: $(which python3.9)"
echo "========================="

# ----------------------------------------------------------------------------
# STEP 1: Aggregate individual splits (Create .npz files)
# ----------------------------------------------------------------------------
echo ""
echo ">>> STEP 1: Aggregating chunks for each split..."

for i in $(seq 0 $((N_SPLITS - 1))); do
    SPLIT_SEED=$((SPLIT_SEED_START + i))
    
    echo ""
    echo ">>> Aggregating Split ${i}/${N_SPLITS}: Seed=${SPLIT_SEED}"
    
    python3.9 ${SCRIPT} \
      --dataset adult \
      --model rf \
      --output_dir ${OUTPUT_DIR} \
      --mode aggregate_split \
      --split_seed ${SPLIT_SEED} \
      --seed 42
    
    echo "✓ Split ${SPLIT_SEED} aggregated successfully"
done

# ----------------------------------------------------------------------------
# STEP 2: Universal Clustering & Disagreement (Generate CSV)
# ----------------------------------------------------------------------------
echo ""
echo ">>> STEP 2: Running Universal Clustering & Disagreement Analysis..."

# Construct comma-separated string of seeds (e.g., "100,101,102,103,104")
SEEDS=$(seq -s, $SPLIT_SEED_START $((SPLIT_SEED_START + N_SPLITS - 1)))

python3.9 ${SCRIPT} \
  --dataset adult \
  --model rf \
  --output_dir ${OUTPUT_DIR} \
  --mode aggregate_universal \
  --split_seeds "${SEEDS}" \
  --disagreement \
  --seed 42

echo ""
echo "========================================================================"
echo "✓ Aggregation Complete!"
echo "========================================================================"
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Generated files:"
ls -lh "${OUTPUT_DIR}" | grep -E "(universal|disagreement|\.npz)" || echo "Warning: Expected files not found"
echo ""
echo "Key files to check:"
echo "  - universal_summary.json (entropy, cluster info)"
echo "  - universal_labels.npy (cluster assignments)"
echo "  - disagreement_report_split*.csv (prediction disagreements)"
echo "========================================================================"
