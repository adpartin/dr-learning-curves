#!/bin/bash

# Example:
# lc_keras.bash gdsc nn_reg0 0
# lc_keras.bash gdsc2 nn_reg0 0

# Call this function from the main project dir!
OUTDIR=lc.out
mkdir -p $OUTDIR
echo "Outdir $OUTDIR"

# LC_SIZES=5
LC_SIZES=7
# LC_SIZES=12

# EPOCH=2
# EPOCH=10
EPOCH=500

SPLIT=0

# Default HPs
SOURCE=$1
MODEL=$2
DEVICE=$3

export CUDA_VISIBLE_DEVICES=$3
echo "Source: $SOURCE"
echo "Model:  $MODEL"
echo "CUDA device: $CUDA_VISIBLE_DEVICES"
echo "LC sizes: $LC_SIZES"

data_version=July2020
dpath=data/ml.dfs/$data_version/data.$SOURCE.dd.ge/data.$SOURCE.dd.ge.parquet 
spath=data/ml.dfs/$data_version/data.$SOURCE.dd.ge/data.$SOURCE.dd.ge.splits 
# ps_hpo_dir=k-tuner/${SOURCE}_${MODEL}_tuner_out/ps_hpo
ls_hpo_dir=k-tuner/${SOURCE}_${MODEL}_tuner_out/ls_hpo

n_runs=1
# n_runs=3
for r in $(seq 1 $n_runs); do
    echo "Run $r"

    # r=7
    python src/main_lc.py \
        -dp $dpath \
        -sd $spath \
        --split_id $SPLIT \
        --ml $MODEL \
        --epoch $EPOCH \
        --batchnorm \
        --gout $OUTDIR/lc.${SOURCE}.${MODEL}.ls_hpo \
        --ls_hpo_dir $ls_hpo_dir \
        --rout run$r \
        --min_size 20000 \
        --lc_sizes $LC_SIZES

        # --lc_sizes_arr 700000 500000 

        # --lc_sizes $LC_SIZES \
        # --min_size 10000 

        # --max_size 700000
        # --lc_sizes_arr 10000 20000 50000 100000 300000 500000 700000

        # --lc_sizes_arr 400000
        # --ps_hpo_dir $ps_hpo_dir \
        # --lc_sizes $LC_SIZES \
        # --min_size 2024
done

