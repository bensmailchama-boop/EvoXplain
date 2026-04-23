#!/bin/tcsh
#SBATCH --job-name=evoxplain_test
#SBATCH --partition=core32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ======================================================================
# EvoXplain Falsification Battery — Generic SLURM Launcher
# ======================================================================
# Usage:
#   sbatch --job-name=tcga_test1 evoxplain_test.sbatch \
#       tcga_test1_determinism.py --mode all --output_dir results/test1
#
#   sbatch --job-name=tcga_test2 evoxplain_test.sbatch \
#       tcga_test2_null_noise.py --output_dir results/test2
#
#   sbatch --job-name=tcga_test7 evoxplain_test.sbatch \
#       tcga_test7_data_degeneracy.py --output_dir results/test7
#
# The script name and all arguments are passed via $argv.
# ======================================================================

mkdir -p logs

cd /home/chamabens/evoxplain

echo "============================================================"
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST  CPUs: $SLURM_CPUS_PER_TASK"
echo "Script: $argv"
echo "Started: `date`"
echo "============================================================"

/usr/bin/python -u $argv

echo ""
echo "============================================================"
echo "Finished: `date`"
echo "============================================================"
