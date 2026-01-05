#!/bin/bash
#SBATCH --job-name=bc_variedC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/bc_variedC_out_%A_%a.log
#SBATCH --error=logs/bc_variedC_err_%A_%a.log
#SBATCH --array=0-499

# =============================================================================
# USER CONFIGURATION - MODIFY THESE FOR YOUR ENVIRONMENT
# =============================================================================
# Set your working directory (where EvoXplain code lives)
WORKDIR="/path/to/your/EvoXplain"  # <-- CHANGE THIS

# Set your partition name
#SBATCH --partition=your_partition  # <-- CHANGE THIS

# Set Python command (may be python3, python3.9, etc.)
PYTHON="python3"  # <-- CHANGE IF NEEDED
# =============================================================================

cd "${WORKDIR}" || exit 1

set -euo pipefail

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
N_SPLITS=10
SPLIT_SEED_START=100
N_CHUNKS=50
N_RUNS_TOTAL=1000
RUNS_PER_CHUNK=20
C_MIN=1e-2
C_MAX=1e2

SCRIPT="evoxplain_core_engine.py"
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: $SCRIPT not found in ${WORKDIR}!"
    exit 1
fi

# =============================================================================
# COMPUTE SPLIT AND CHUNK INDICES
# =============================================================================
# Array task 0-499 maps to:
#   - 10 splits (0-9) × 50 chunks (0-49) = 500 jobs
SPLIT_OFFSET=$((SLURM_ARRAY_TASK_ID / N_CHUNKS))
CHUNK_INDEX=$((SLURM_ARRAY_TASK_ID % N_CHUNKS))
SPLIT_SEED=$((SPLIT_SEED_START + SPLIT_OFFSET))

OUTPUT_DIR="results/bc_lr_shap_variedC"

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
echo "C_MODE=varied [${C_MIN}, ${C_MAX}]"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "===================="

# =============================================================================
# RUN EVOXPLAIN
# =============================================================================
${PYTHON} ${SCRIPT} \
  --dataset breast_cancer \
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
