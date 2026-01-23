#!/bin/bash
#SBATCH --job-name=compas_variedC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/compas_variedC_out_%A_%a.log
#SBATCH --error=logs/compas_variedC_err_%A_%a.log
#SBATCH --partition=core32
#SBATCH --array=0-499
#SBATCH --chdir=/home2/chamabens/HPC_EvoXplain

set -euo pipefail

# =============================================================================
# CONFIGURATION - COMPAS Varied C Experiment
# =============================================================================
N_SPLITS=10
SPLIT_SEED_START=100
N_CHUNKS=50
N_RUNS_TOTAL=1000
RUNS_PER_CHUNK=20
C_MIN=1e-2
C_MAX=1e2

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
OUTPUT_DIR="results/compas_lr_shap_variedC"

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
echo "DATASET=compas"
echo "SPLIT_SEED=${SPLIT_SEED}"
echo "CHUNK_INDEX=${CHUNK_INDEX} (--chunk_id)"
echo "RUNS_PER_CHUNK=${RUNS_PER_CHUNK} (--chunk_size)"
echo "N_RUNS_TOTAL=${N_RUNS_TOTAL}"
echo "C_MODE=varied [${C_MIN}, ${C_MAX}]"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "===================="

# =============================================================================
# RUN EVOXPLAIN
# =============================================================================
python3.9 ${SCRIPT} \
  --dataset compas \
  --model logreg \
  --mode chunk \
  --split_seed ${SPLIT_SEED} \
  --n_runs ${N_RUNS_TOTAL} \
  --chunk_id ${CHUNK_INDEX} \
  --chunk_size ${RUNS_PER_CHUNK} \
  --c_mode varied \
  --c_min ${C_MIN} \
  --c_max ${C_MAX} \
  --output_dir ${OUTPUT_DIR}

echo ""
echo "=== POSTRUN: Files created ==="
ls -lh "${OUTPUT_DIR}"/importance_split${SPLIT_SEED}_chunk${CHUNK_INDEX}.npy 2>/dev/null || echo "(importance file not found)"
ls -lh "${OUTPUT_DIR}"/meta_split${SPLIT_SEED}_chunk${CHUNK_INDEX}.json 2>/dev/null || echo "(meta file not found)"
echo "=== DONE ==="
