#!/bin/bash -l
#SBATCH --job-name=surrogate_LD18
#SBATCH --partition=CPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-239%4
#SBATCH --output=surrogate_%A_%a.out
#SBATCH --error=surrogate_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fenicsx-env-312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u src/rve_simulation/generate_surrogate_dataset.py \
  --config configs/surrogate_dataset_v1.json \
  --task-id "${SLURM_ARRAY_TASK_ID}"
