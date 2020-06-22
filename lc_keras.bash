#!/bin/bash

OUTDIR=lc.out
mkdir -p $OUTDIR
echo "Outdir $OUTDIR"

# lc_sizes=12
# lc_sizes=5
lc_sizes=7

# EPOCH=400
EPOCH=500
# EPOCH=2

# Default HPs
SOURCE=$1
DEVICE=$2
MODEL=$3
export CUDA_VISIBLE_DEVICES=$2
echo "Processing source: $SOURCE"
echo "CUDA device: $CUDA_VISIBLE_DEVICES"
echo "LC sizes: $lc_sizes"

# n_runs=3
# for r in $(seq 1 $n_runs); do
#     echo "Run $r"

#     # r=7
#     python src/main_lc.py \
#         -dp data/ml.dfs/data.$SOURCE.dd.ge.raw/data.$SOURCE.dd.ge.raw.parquet \
#         -sd data/ml.dfs/data.$SOURCE.dd.ge.raw/data.$SOURCE.dd.ge.raw.splits \
#         --split_id 0 \
#         --fea_prfx ge dd --fea_sep _ \
#         -t AUC -sc stnd --ml $MODEL \
#         --batch_size 64 --epoch $EPOCH \
#         --batchnorm \
#         --gout $OUTDIR/lc.${SOURCE}.${MODEL}.test_fair \
#         --rout run$r \
#         --lc_sizes $lc_sizes

#     # --lc_sizes_arr 88416 \

# done


n_runs=3
# n_runs=1
for r in $(seq 1 $n_runs); do
    echo "Run $r"

    # r=7
    python src/main_lc.py \
        -dp data/ml.dfs/data.$SOURCE.dd.ge.raw/data.$SOURCE.dd.ge.raw.parquet \
        -sd data/ml.dfs/data.$SOURCE.dd.ge.raw/data.$SOURCE.dd.ge.raw.splits \
        --split_id 0 \
        --fea_prfx ge dd --fea_sep _ \
        -t AUC -sc stnd --ml $MODEL \
        --batch_size 32 --epoch $EPOCH \
        --batchnorm \
        --gout $OUTDIR/lc.${SOURCE}.${MODEL}.ps_hpo.test_fair \
        --ps_hpo_dir k-tuner/${SOURCE}_${MODEL}_tuner_out/ps_hpo \
        --rout run$r \
        --lc_sizes $lc_sizes \
        --min_size 2024

        # --lc_sizes_arr 88416 \

done

