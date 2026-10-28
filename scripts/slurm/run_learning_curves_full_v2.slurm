#!/bin/bash -l
#SBATCH --partition=CPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=learning_full_v2
#SBATCH --array=0-134%2
#SBATCH --output=learning_full_v2_%A_%a.out
#SBATCH --error=learning_full_v2_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fenicsx-env-312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u src/surrogate_modeling/run_surrogate_supplement_v2.py \
  --config configs/surrogate_supplement_v2.json \
  --mode learning --task-id "${SLURM_ARRAY_TASK_ID}" \
  --n-jobs "${SLURM_CPUS_PER_TASK}"
