#!/bin/bash

# For example:
# bash tuner.bash gdsc 10000 nn0
SOURCE=$1
TR_SIZE=$2
MODEL=$3

# export CUDA_VISIBLE_DEVICES=1
echo "Source: $SOURCE"
echo "Model: $MODEL"
echo "Training size: $TR_SIZE"
# echo "CUDA device: $CUDA_VISIBLE_DEVICES"
python k-tuner/my-tuner.py --source $SOURCE --seed 0 --sz $TR_SIZE --ml $MODEL
