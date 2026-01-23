#!/bin/bash
#SBATCH --job-name=bc_rf_universal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/bc_rf_universal_%j.log
#SBATCH --error=logs/bc_rf_universal_%j.err
#SBATCH --partition=core32
#SBATCH --chdir=/home2/chamabens/HPC_EvoXplain

# ============================================================================
# Universal Aggregation + Disagreement Analysis for BC RF Varied
# This combines all 5 splits and performs clustering + disagreement inspection
# ============================================================================

set -euo pipefail

# --- Configuration ---
SPLIT_SEEDS="100,101,102,103,104"
OUTPUT_DIR="results/bc_rf_shap_varied"

# --- Point to the Script ---
SCRIPT="evoxplain_core_engine.py"
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: $SCRIPT not found!"
    exit 1
fi

# --- Environment check ---
echo "=== Environment Check ==="
echo "Using Script: $SCRIPT"
which python3.9
python3.9 --version
python3.9 -c "import shap; import sklearn; print('Packages OK')" || { echo "ERROR: Missing packages"; exit 1; }
echo "========================="
echo ""

mkdir -p logs

echo "========================================================================"
echo "Running Universal Aggregation + Disagreement Analysis"
echo "Split seeds: ${SPLIT_SEEDS}"
echo "========================================================================"

python3.9 ${SCRIPT} \
  --dataset breast_cancer \
  --model rf \
  --mode aggregate_universal \
  --split_seeds ${SPLIT_SEEDS} \
  --output_dir ${OUTPUT_DIR} \
  --disagreement \
  --rf_varied

echo ""
echo "========================================================================"
echo "Universal aggregation complete!"
echo "Check for disagreement_report_split*.csv in ${OUTPUT_DIR}"
echo "========================================================================"

ls -la ${OUTPUT_DIR}/*.csv 2>/dev/null || echo "No CSV files found yet"
ls -la ${OUTPUT_DIR}/universal* 2>/dev/null || echo "No universal files found yet"
