#!/bin/bash -l
#SBATCH --job-name=surrogate_preprocess_v2
#SBATCH --partition=CPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=surrogate_preprocess_v2_%j.out
#SBATCH --error=surrogate_preprocess_v2_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fenicsx-env-312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u src/surrogate_modeling/audit_microstructure_descriptors_v2.py
python -u src/surrogate_modeling/generate_two_point_statistics_v2.py
python -u src/surrogate_modeling/normalize_two_point_statistics_v2.py
python -u src/surrogate_modeling/check_surrogate_preprocessing_v2.py
