#!/bin/bash -l
#SBATCH --job-name=surrogate_ext_v1
#SBATCH --partition=CPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-199%4
#SBATCH --output=surrogate_ext_%A_%a.out
#SBATCH --error=surrogate_ext_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fenicsx-env-312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u src/rve_simulation/generate_surrogate_dataset_extension.py \
  --config configs/surrogate_dataset_extension_v1.json \
  --task-id "${SLURM_ARRAY_TASK_ID}"
