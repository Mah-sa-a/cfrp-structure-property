#!/bin/bash -l
#SBATCH --job-name=packing_v2
#SBATCH --partition=CPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-47%4
#SBATCH --output=packing_v2_%A_%a.out
#SBATCH --error=packing_v2_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fenicsx-env-312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -c 'import sys,gmsh,dolfinx,dolfinx_mpc; print("Python:",sys.executable,"Gmsh:",gmsh.__file__,flush=True)'
python -u src/rve_simulation/packing_study_v2.py --config configs/packing_study_v2.json --task-id "${SLURM_ARRAY_TASK_ID}"
