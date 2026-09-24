#!/bin/bash -l

### HPCassiopee_V3.0

### This script is to be used for GP/GPGPU computations
#
## HPCassiopee GPU Servers : 
#  gpu1 : - server : CPUs=2*12 | 192 GB RAM
#         - device 1    : V100S * 1 | 1*32 GB VRAM 
#         - device[2-3] : T4    * 2 | 1*16 GB VRAM
#
#  gpu3 : - server :  CPUs=2*16 | 192 GB RAM
#         - device[1-4] : K80   : 4 | 6*8GB  VRAM 
#
## When allocating GPUs, use the "#SBATCH --gres=gpu:<TYPE>:<NUMBER>
## where TYPE:NUMBER is one of :
##  - V100S:1
##  - K80:[1-4]
##  - T4:[1-2]

## To check correct usage of gpus: ssh gpu1 and nvtop (example for node "gpu1")

### User fill in HERE
#
## Required batch arguments
#SBATCH --job-name=SAM
#SBATCH --partition=GPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#
## Optional batch arguments (uncomment if used)
##SBATCH --time=100:00:00
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mahsa.arabkoohichetabi@parisnanterre.fr
#
## Suggested batch arguments 
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err


### Module Selection, send 'module avail' for a list
module purge
module load EasyBuild/V1
module load Anaconda3/2024.02-1
 
# A lot of python packages available! try "module load EasyBuild foss" and "module available" in a terminal
# "module help SciPy-bundle" to get the list of provided packages by SciPy-bundle: numpy, scipy, panda, mpi4py...

### Activate Conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fenicsx-env

### Commands
#

cd "$SLURM_SUBMIT_DIR"
python homogenize_fibre_RVEs_fenicsx.py

### EOF
