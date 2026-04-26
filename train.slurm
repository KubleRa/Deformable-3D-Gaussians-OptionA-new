#!/bin/bash
# Changed — Option A: single-GPU SLURM submission for NeuMan training.
# Submit with:
#   sbatch train.slurm
# Override the scene without editing the file:
#   sbatch --export=ALL,SCENE=citron train.slurm

#SBATCH -A DMS26009                     # Project / allocation
#SBATCH -J OptA-train                   # Job name
#SBATCH -o logs/OptA-%x.o%j             # stdout (logs/ created below)
#SBATCH -e logs/OptA-%x.e%j             # stderr
#SBATCH -p gpu-a100-small               # Queue (GPU partition)
#SBATCH -N 1                            # 1 node
#SBATCH -n 1                            # 1 task
#SBATCH -t 01:00:00                     # hh:mm:ss — bump for longer runs
# #SBATCH --mail-type=all
# #SBATCH --mail-user=rak3284@my.utexas.edu

set -euo pipefail

# ---- paths (edit here if layout changes) -------------------------------------
CODE_DIR=$WORK/Deformable-3D-Gaussians-Alex
# VENV=/scratch/11293/rak3284/venvs/GaussianSplattingEnv        # adjust if your venv lives elsewhere
SCENE="${SCENE:-standup}"                      # override via --export=ALL,SCENE=...
# DATA_ROOT=/scratch/11293/rak3284/ls6/D-NeuMan/neuman/dataset
DATA_ROOT=$SCRATCH/ls6/data
OUT_ROOT=$WORK/Deformable-3D-Gaussians-Alex
# ------------------------------------------------------------------------------

mkdir -p logs "$OUT_ROOT"

# Toolchain: GCC 11 (for GLIBCXX_3.4.29) + CUDA 11.6 to match the venv build.
module load gcc/11.2.0 || true
#module load cuda/11.6  || true
module list

# source "$VENV/bin/activate"
source $SCRATCH/venvs/GaussianSplattingEnv/bin/activate

echo "Host:       $(hostname)"
echo "Scene:      $SCENE"
echo "Python:     $(which python)"
nvidia-smi || true

cd "$CODE_DIR"
date
bash ./gausstrain.sh "$SCENE"
date
