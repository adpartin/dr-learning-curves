#!/bin/bash

# For example:
# bash tuner.bash gdsc 10000 nn0
SOURCE=$1
TR_SIZE=$2
VL_SIZE=$3
MODEL=$4

# DEVICE=$5
# export CUDA_VISIBLE_DEVICES=$DEVICE
# echo "CUDA device: $CUDA_VISIBLE_DEVICES"

echo "Source: $SOURCE"
echo "Model: $MODEL"
echo "Training size: $TR_SIZE"
echo "Validation size: $VL_SIZE"

python k-tuner/my-tuner.py --source $SOURCE --seed 0 --tr_sz $TR_SIZE --vl_sz $VL_SIZE --ml $MODEL
