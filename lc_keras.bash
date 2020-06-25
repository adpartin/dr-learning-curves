#!/bin/bash

OUTDIR=lc.out
mkdir -p $OUTDIR
echo "Outdir $OUTDIR"

# LC_SIZES=5
LC_SIZES=7
# LC_SIZES=12

# EPOCH=2
EPOCH=500

SPLIT=0

# Default HPs
SOURCE=$1
DEVICE=$2
MODEL=$3
export CUDA_VISIBLE_DEVICES=$2
echo "Processing source: $SOURCE"
echo "CUDA device: $CUDA_VISIBLE_DEVICES"
echo "LC sizes: $lc_sizes"

dpath=data/ml.dfs/data.$SOURCE.dd.ge.raw/data.$SOURCE.dd.ge.raw.parquet 
spath=data/ml.dfs/data.$SOURCE.dd.ge.raw/data.$SOURCE.dd.ge.raw.splits 

n_runs=3
for r in $(seq 1 $n_runs); do
    echo "Run $r"

    # r=7
    python src/main_lc.py \
        -dp $dpath \
        -sd $spath \
        --split_id $SPLIT \
        --fea_prfx ge dd --fea_sep _ \
        -t AUC -sc stnd --ml $MODEL \
        --batch_size 32 --epoch $EPOCH \
        --batchnorm \
        --gout $OUTDIR/lc.${SOURCE}.${MODEL}.ps_hpo.test_fair \
        --ps_hpo_dir k-tuner/${SOURCE}_${MODEL}_tuner_out/ps_hpo \
        --rout run$r \
        --lc_sizes $LC_SIZES \
        --min_size 2024

        # --lc_sizes_arr 88416 \
done

