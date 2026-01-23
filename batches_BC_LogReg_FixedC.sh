#!/bin/bash
#SBATCH --job-name=bc_fixedC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/bc_fixedC_out_%A_%a.log
#SBATCH --error=logs/bc_fixedC_err_%A_%a.log
#SBATCH --partition=core32
#SBATCH --array=0-499
#SBATCH --chdir=/home2/chamabens/HPC_EvoXplain

set -euo pipefail

# =============================================================================
# CONFIGURATION - Fixed C=1.0 Experiment
# =============================================================================
N_SPLITS=10
SPLIT_SEED_START=100
N_CHUNKS=50
N_RUNS_TOTAL=1000
RUNS_PER_CHUNK=20
FIXED_C_VALUE=1.0

# Script selection
SCRIPT="evoxplain_core_engine.py"
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: $SCRIPT not found!"
    exit 1
fi

# =============================================================================
# COMPUTE SPLIT AND CHUNK INDICES
# =============================================================================
SPLIT_OFFSET=$((SLURM_ARRAY_TASK_ID / N_CHUNKS))
CHUNK_INDEX=$((SLURM_ARRAY_TASK_ID % N_CHUNKS))
SPLIT_SEED=$((SPLIT_SEED_START + SPLIT_OFFSET))

# Output directory
OUTPUT_DIR="results/bc_lr_shap_fixedC_C${FIXED_C_VALUE}"

mkdir -p logs "${OUTPUT_DIR}"

# =============================================================================
# DEBUG INFO
# =============================================================================
echo "=== JOB INFO ==="
echo "HOST=$(hostname)"
echo "CWD=$(pwd)"
echo "SCRIPT=${SCRIPT}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo ""
echo "=== EVOXPLAIN PARAMETERS ==="
echo "SPLIT_SEED=${SPLIT_SEED}"
echo "CHUNK_INDEX=${CHUNK_INDEX} (--chunk_id)"
echo "RUNS_PER_CHUNK=${RUNS_PER_CHUNK} (--chunk_size)"
echo "N_RUNS_TOTAL=${N_RUNS_TOTAL}"
echo "C_MODE=fixed, C=${FIXED_C_VALUE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "===================="

# =============================================================================
# RUN EVOXPLAIN
# =============================================================================
# FIXED: Use --chunk_id and --chunk_size instead of --chunk_start/--chunk_end
# FIXED: Removed --tag (not supported)
# FIXED: Use --C instead of --c_fixed
# FIXED: Added --model logreg

python3.9 ${SCRIPT} \
  --dataset breast_cancer \
  --model logreg \
  --mode chunk \
  --split_seed ${SPLIT_SEED} \
  --n_runs ${N_RUNS_TOTAL} \
  --chunk_id ${CHUNK_INDEX} \
  --chunk_size ${RUNS_PER_CHUNK} \
  --c_mode fixed \
  --C ${FIXED_C_VALUE} \
  --output_dir ${OUTPUT_DIR}

echo ""
echo "=== POSTRUN: Files created ==="
ls -lh "${OUTPUT_DIR}"/importance_split${SPLIT_SEED}_chunk${CHUNK_INDEX}.npy 2>/dev/null || echo "(importance file not found)"
ls -lh "${OUTPUT_DIR}"/meta_split${SPLIT_SEED}_chunk${CHUNK_INDEX}.json 2>/dev/null || echo "(meta file not found)"
echo "=== DONE ==="
