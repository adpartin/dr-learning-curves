#!/bin/bash

# For example:
# bash tuner.bash gdsc 10000 nn0

# bash tuner.bash gdsc2 lgb 0.2

SOURCE=$1
MODEL=$2
VL_SIZE=$3
# TR_SIZE=$4

# DEVICE=$5
# export CUDA_VISIBLE_DEVICES=$DEVICE
# echo "CUDA device: $CUDA_VISIBLE_DEVICES"

echo "Source:     $SOURCE"
echo "Model:      $MODEL"
echo "Val size:   $VL_SIZE"
# echo "Train size: $TR_SIZE"

python k-tuner/my-tuner.py \
    --source $SOURCE \
    --seed 0 \
    --vl_sz $VL_SIZE \
    --ml $MODEL

    # --tr_sz $TR_SIZE \
