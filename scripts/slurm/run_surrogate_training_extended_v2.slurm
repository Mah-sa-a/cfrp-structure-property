#!/bin/bash -l
#SBATCH --job-name=extended_v2
#SBATCH --partition=CPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=extended_v2_%j.out
#SBATCH --error=extended_v2_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fenicsx-env-312
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
python -u src/surrogate_modeling/train_surrogate_extended_v2.py \
  --config configs/surrogate_training_extended_v2.json \
  --stage train
