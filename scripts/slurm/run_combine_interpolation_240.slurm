#!/bin/bash -l
#SBATCH --job-name=combine_interp_240
#SBATCH --partition=CPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=combine_interp_240_%j.out
#SBATCH --error=combine_interp_240_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fenicsx-env-312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u src/surrogate_modeling/combine_interpolation_240.py
