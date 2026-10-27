#!/bin/bash -l
#SBATCH --job-name=replace_0084
#SBATCH --partition=CPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=recover_0084_%j.out
#SBATCH --error=recover_0084_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module load EasyBuild/V1
module load Anaconda3/2024.02-1

CONDA_BIN="/modules/EasyBuild/V1/software/Anaconda3/2024.02-1/bin/conda"
ENV_DIR="/home/BeeGFS/ExtLabs/LEME/marabkoohichetabi/.conda/envs/fenicsx-env-312"

test -x "${CONDA_BIN}"
test -x "${ENV_DIR}/bin/python"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

"${CONDA_BIN}" run --no-capture-output -p "${ENV_DIR}" \
  "${ENV_DIR}/bin/python" -c \
  'import sys, gmsh, dolfinx, dolfinx_mpc; print("Python:", sys.executable, flush=True); print("Gmsh:", gmsh.__file__, flush=True); print("Imports passed", flush=True)'

"${CONDA_BIN}" run --no-capture-output -p "${ENV_DIR}" \
  "${ENV_DIR}/bin/python" -u \
  src/rve_simulation/recover_sample_0084.py --run --seed 30084
