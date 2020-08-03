#!/bin/bash

# Example:
# lc_keras.bash nci60 nn_reg0 0 flatten
# lc_keras.bash nci60 nn_reg0 0 random

OUTDIR=lc.out_02
mkdir -p $OUTDIR
echo "Outdir $OUTDIR"

LC_SIZES=7
# LC_SIZES=12

# EPOCH=2
EPOCH=500

SPLIT=0

SOURCE=$1
MODEL=$2
DEVICE=$3
SAMPLING=$4

export CUDA_VISIBLE_DEVICES=$3
echo "Source: $SOURCE"
echo "Model:  $MODEL"
echo "CUDA:   $CUDA_VISIBLE_DEVICES"
echo "SAMPLING: $SAMPLING"

data_version=July2020
dpath=data/ml.dfs/$data_version/data.$SOURCE.dd.ge.$SAMPLING/data.$SOURCE.dd.ge.parquet 
spath=data/ml.dfs/$data_version/data.$SOURCE.dd.ge.$SAMPLING/data.$SOURCE.dd.ge.splits 
ls_hpo_dir=k-tuner/${SOURCE}_${MODEL}_tuner_out/ls_hpo

r=7
python src/main_lc.py \
    -dp $dpath \
    -sd $spath \
    --split_id $SPLIT \
    --ml $MODEL \
    --epoch $EPOCH \
    --batchnorm \
    --gout $OUTDIR/lc.${SOURCE}.${MODEL}.${SAMPLING}.ls_hpo \
    --ls_hpo_dir $ls_hpo_dir \
    --rout run$r \
    --lc_sizes_arr 500000 570000 640000

#     --min_size 20000 \
#     --lc_sizes $LC_SIZES
