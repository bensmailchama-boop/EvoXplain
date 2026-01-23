#!/bin/bash
#SBATCH --job-name=bc_rf_varied
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/bc_rf_varied_out_%A_%a.log
#SBATCH --error=logs/bc_rf_varied_err_%A_%a.log
#SBATCH --partition=core32
#SBATCH --array=0-249
#SBATCH --chdir=/home2/chamabens/HPC_EvoXplain

# ============================================================================
# Breast Cancer — Random Forest (Varied Hyperparams)
# 5 splits * 50 chunks = 250 tasks (Array 0-249)
# ============================================================================

set -euo pipefail

# --- Configuration ---
N_SPLITS=5
SPLIT_SEED_START=100
N_CHUNKS=50
N_RUNS_TOTAL=1000
CHUNK_SIZE=20   # 1000 runs / 50 chunks = 20 runs per chunk

# --- Point to the Script ---
SCRIPT="evoxplain_core_engine.py"
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: $SCRIPT not found!"
    exit 1
fi

# --- Compute split seed and chunk index from array task ID ---
SPLIT_OFFSET=$((SLURM_ARRAY_TASK_ID / N_CHUNKS))
CHUNK_ID=$((SLURM_ARRAY_TASK_ID % N_CHUNKS))
SPLIT_SEED=$((SPLIT_SEED_START + SPLIT_OFFSET))

# --- Output directory ---
OUTPUT_DIR="results/bc_rf_shap_varied"

mkdir -p logs "${OUTPUT_DIR}"

echo "=== DEBUG ==="
echo "HOST=$(hostname)"
echo "CWD=$(pwd)"
echo "SCRIPT=${SCRIPT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "SPLIT_SEED=${SPLIT_SEED}  CHUNK_ID=${CHUNK_ID}  CHUNK_SIZE=${CHUNK_SIZE}"
echo "=============="

# --- Run Python Script (RF Varied Mode) ---
python3.9 ${SCRIPT} \
  --dataset breast_cancer \
  --model rf \
  --mode chunk \
  --split_seed ${SPLIT_SEED} \
  --n_runs ${N_RUNS_TOTAL} \
  --chunk_id ${CHUNK_ID} \
  --chunk_size ${CHUNK_SIZE} \
  --output_dir ${OUTPUT_DIR} \
  --rf_varied

echo "=== POSTRUN LS (${OUTPUT_DIR}) ==="
ls -lh "${OUTPUT_DIR}" | tail -n 20 || true
