#!/bin/bash
#SBATCH --job-name=adult_rf_varied
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/adult_rf_varied_out_%A_%a.log
#SBATCH --error=logs/adult_rf_varied_err_%A_%a.log
#SBATCH --partition=core32
#SBATCH --array=0-249
#SBATCH --chdir=/home2/chamabens/HPC_EvoXplain

# ============================================================================
# Adult Income — Random Forest (Varied Hyperparams)
# 5 splits * 50 chunks = 250 tasks (Array 0-249)
# ============================================================================

set -euo pipefail

# --- Configuration ---
N_SPLITS=5
SPLIT_SEED_START=100
N_CHUNKS=50
N_RUNS_TOTAL=1000
RUNS_PER_CHUNK=20

# --- Point to the Correct Core Engine Script ---
SCRIPT="evoxplain_core_engine.py"
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: $SCRIPT not found. Please ensure the updated python script is in the directory."
    exit 1
fi

# --- Verify Adult Dataset Exists ---
ADULT_DATA="data/adult.csv"
if [ ! -f "$ADULT_DATA" ]; then
    echo "ERROR: Adult dataset not found at ${ADULT_DATA}"
    echo "Please run: bash download_adult_data.sh"
    exit 1
fi

# --- Compute split seed and chunk index from array task ID ---
SPLIT_OFFSET=$((SLURM_ARRAY_TASK_ID / N_CHUNKS))
CHUNK_INDEX=$((SLURM_ARRAY_TASK_ID % N_CHUNKS))
SPLIT_SEED=$((SPLIT_SEED_START + SPLIT_OFFSET))

# --- Output directory ---
OUTPUT_DIR="results/adult_rf_shap_varied"

mkdir -p logs "${OUTPUT_DIR}"

echo "=== DEBUG ==="
echo "HOST=$(hostname)"
echo "CWD=$(pwd)"
echo "SCRIPT=${SCRIPT}"
echo "ADULT_DATA=${ADULT_DATA}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "SPLIT_SEED=${SPLIT_SEED}  CHUNK_INDEX=${CHUNK_INDEX}"
echo "RUNS: ${RUNS_PER_CHUNK} per chunk, ${N_RUNS_TOTAL} total"
echo "=============="

# --- Run Python Script (RF Varied Mode) ---
python3.9 ${SCRIPT} \
  --dataset adult \
  --model rf \
  --mode chunk \
  --split_seed ${SPLIT_SEED} \
  --n_runs ${N_RUNS_TOTAL} \
  --chunk_id ${CHUNK_INDEX} \
  --chunk_size ${RUNS_PER_CHUNK} \
  --output_dir ${OUTPUT_DIR} \
  --rf_varied \
  --seed 42

echo "=== CHUNK ${CHUNK_INDEX} COMPLETE (Split ${SPLIT_SEED}) ==="

echo "=== POSTRUN LS (${OUTPUT_DIR}) ==="
ls -lh "${OUTPUT_DIR}" | tail -n 20 || true
