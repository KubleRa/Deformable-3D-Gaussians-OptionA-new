#!/bin/bash
# Changed — Option A: thin wrapper around `python train.py` for NeuMan scenes.
# Called by train.slurm, but can also be run directly on a GPU node:
#   bash gausstrain.sh bike
# First arg (optional) is the scene name; defaults to $SCENE, then "bike".

set -euo pipefail

SAVE_PATH=output/run256
HEAD_LAYER=256

SCENE="${SCENE:-standup}" 

DATA_ROOT=$SCRATCH/ls6/data
OUT_ROOT=$WORK/Deformable-3D-Gaussians-Alex

DATA_ROOT=/scratch/11293/rak3284/ls6/D-NeuMan/neuman/dataset
OUT_ROOT=/work/11293/rak3284/ls6/outputs/optionA/one_seed/r2

SRC="$DATA_ROOT/$SCENE"
OUT="$OUT_ROOT/$SCENE/$SAVE_PATH"

if [ ! -d "$SRC" ]; then
    echo "ERROR: scene dir not found: $SRC" >&2
    exit 1
fi
mkdir -p "$OUT"

echo "Training scene '$SCENE' (Option A)"
echo "  source: $SRC"
echo "  output: $OUT"echo
python train.py \
    --source_path "$SRC" \
    --model_path  "$OUT" \
    --eval \
    --head_layer=$HEAD_LAYER
