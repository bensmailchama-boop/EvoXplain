#!/bin/bash
#SBATCH --job-name=adult_variedC_chunk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/adult_variedC_%A_%a.log
#SBATCH --error=logs/adult_variedC_%A_%a.err
#SBATCH --partition=core32
#SBATCH --chdir=/home2/chamabens/HPC_EvoXplain
#SBATCH --array=0-99

set -euo pipefail

# =============================================================================
# CONFIGURATION - Varied C Experiment (Adult Dataset)
# =============================================================================
N_SPLITS=10
SPLIT_SEED_START=100
N_RUNS=1000
CHUNK_SIZE=100
C_MIN=0.01
C_MAX=100.0
OUTPUT_DIR="results/adult_lr_shap_variedC"
SCRIPT="evoxplain_core_engine.py"

# Total chunks = N_SPLITS * (N_RUNS / CHUNK_SIZE) = 10 * 10 = 100
N_CHUNKS_PER_SPLIT=$((N_RUNS / CHUNK_SIZE))

mkdir -p logs
mkdir -p ${OUTPUT_DIR}

# Decode array task ID into split and chunk
SPLIT_IDX=$((SLURM_ARRAY_TASK_ID / N_CHUNKS_PER_SPLIT))
CHUNK_ID=$((SLURM_ARRAY_TASK_ID % N_CHUNKS_PER_SPLIT))
SPLIT_SEED=$((SPLIT_SEED_START + SPLIT_IDX))

echo "=== CHUNK JOB (Adult Dataset, Varied C) ==="
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "SPLIT_IDX=${SPLIT_IDX}, SPLIT_SEED=${SPLIT_SEED}"
echo "CHUNK_ID=${CHUNK_ID}"
echo "HOST=$(hostname)"
echo "C range: [${C_MIN}, ${C_MAX}] (log-uniform)"
echo ""

python3.9 ${SCRIPT} \
  --dataset adult \
  --model logreg \
  --mode chunk \
  --split_seed ${SPLIT_SEED} \
  --n_runs ${N_RUNS} \
  --chunk_id ${CHUNK_ID} \
  --chunk_size ${CHUNK_SIZE} \
  --c_mode varied \
  --c_min ${C_MIN} \
  --c_max ${C_MAX} \
  --output_dir ${OUTPUT_DIR}

echo ""
echo "=== DONE ==="
