#!/bin/bash -l
#SBATCH --job-name=pack_mesh_check
#SBATCH --partition=CPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=pack_mesh_check_%j.out
#SBATCH --error=pack_mesh_check_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fenicsx-env-312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -c 'import sys,gmsh,dolfinx,dolfinx_mpc; print(sys.executable,gmsh.__file__,flush=True)'
python -u src/rve_simulation/packing_mesh_refinement.py --N "${N:?supply N=165 or N=206}" --replicate "${REP:?supply REP=0..11}" --level medium
