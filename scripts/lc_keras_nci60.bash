#!/bin/bash

# Example:
# lc_keras.bash nci60 nn_reg0 0 0 flatten
# lc_keras.bash nci60 nn_reg0 0 0 random

SOURCE=$1
MODEL=$2
DEVICE=$3
SPLIT=$4
SAMPLING=$5

# Call this function from the main project dir!
OUTDIR=lc.out.${SOURCE}.split${SPLIT}
mkdir -p $OUTDIR
echo "Outdir $OUTDIR"

# LC_SIZES=7
LC_SIZES=12

# EPOCH=2
EPOCH=500

export CUDA_VISIBLE_DEVICES=$3
echo "Source: $SOURCE"
echo "Model:  $MODEL"
echo "CUDA:   $CUDA_VISIBLE_DEVICES"
echo "SPLIT:  $SPLIT"
echo "SAMPLING: $SAMPLING"
echo "LC sizes: $LC_SIZES"

data_version=July2020
dpath=data/ml.dfs/$data_version/data.$SOURCE.dd.ge.$SAMPLING/data.$SOURCE.dd.ge.parquet 
spath=data/ml.dfs/$data_version/data.$SOURCE.dd.ge.$SAMPLING/data.$SOURCE.dd.ge.splits 
ls_hpo_dir=k-tuner/${SOURCE}_${MODEL}_tuner_out/ls_hpo

r=1
# r=3
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
    --min_size 100000 \
    --lc_sizes $LC_SIZES

    # --lc_sizes_arr 500001 570000 640000
