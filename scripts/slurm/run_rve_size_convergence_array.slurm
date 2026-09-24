#!/bin/bash -l

### HPCassiopee_V3.0 -- ARRAY-JOB version: runs each (Vf, L_over_D)
### combination as its own SEPARATE, PARALLEL SLURM task.
#
## How the parallelism actually happens:
##   1. #SBATCH --array=0-11 tells SLURM to launch 12 independent
##      copies of this script, differing only in the environment
##      variable SLURM_ARRAY_TASK_ID (0, 1, 2, ..., 11), scheduled to
##      run AT THE SAME TIME across different compute nodes/cores
##      (subject to cluster availability).
##   2. Each of those 12 copies runs "python run_single_combo.py".
##   3. run_single_combo.py reads its own SLURM_ARRAY_TASK_ID, maps it
##      to exactly ONE (Vf, L_over_D) pair out of the full grid, and
##      restricts THAT task's pipeline run to only that one combination.
##   4. Each task writes its own private CSV
##      (results/RVE_size_study/partial_results/task_<id>.csv) so the
##      12 tasks never collide by writing to the same file.
##
## This is different from submit_rve_sweep_HPCassiopee.slurm, which
## runs "python rve_size_convergence_pipeline.py" directly -- that
## single script loops through the ENTIRE grid itself, sequentially,
## inside ONE job. This array script instead lets SLURM run 12 pieces
## of that same grid in parallel.
#
## --array upper bound MUST equal
##   (number of Vf values in VF_LIST) * (number of L_over_D values
##   in L_OVER_D_LIST) - 1.
##
## Current grid in rve_size_convergence_pipeline.py:
##   VF_LIST = [0.40, 0.50, 0.60]        -> 3 values
##   L_OVER_D_LIST = [4, 8, 12, 18, 20, 24]      -> 6 values
##   3 * 6 = 18 combinations -> array=0-17
##
## If you change either list, recompute:
##   new_array_max = len(VF_LIST) * len(L_OVER_D_LIST) - 1

### User fill in HERE
#
## Required batch arguments
#SBATCH --job-name=RVE_sweep_array
#SBATCH --partition=CPU_Compute
#SBATCH --array=0-17
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#
## Optional batch arguments
#SBATCH --time=100:00:00
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mahsa.arabkoohichetabi@parisnanterre.fr
#
## Suggested batch arguments
## %A = the overall array job ID (same for all 12 tasks)
## %a = this specific task's index (0, 1, 2, ..., 11) -- this is what
##      makes each task's log file unique so you can inspect them
##      individually instead of everything overwriting one file.
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err


### Module Selection, send 'module avail' for a list
module purge
module load EasyBuild/V1
module load Anaconda3/2024.02-1

### Activate Conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fenicsx-env-312

### Commands
#
cd "$SLURM_SUBMIT_DIR"

echo "Array task $SLURM_ARRAY_TASK_ID (of array job $SLURM_ARRAY_JOB_ID) "
echo "started on $(hostname) at $(date)"

# THIS is the line that makes each task run only ONE combination --
# run_single_combo.py reads SLURM_ARRAY_TASK_ID internally.
python run_single_combo.py 

echo "Array task $SLURM_ARRAY_TASK_ID finished at $(date)"

# NOTE: do NOT merge or plot here -- other array tasks (different
# SLURM_ARRAY_TASK_ID values) may still be running at this point.
# After confirming ALL 12 tasks have finished (squeue -u $USER shows
# nothing left from this job), run ONCE from the login node:
#
#   python merge_results.py
#   python rve_size_convergence_plots.py

### EOF
