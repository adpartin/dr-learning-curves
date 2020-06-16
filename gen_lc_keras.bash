#!/bin/bash

OUTDIR=lc.out
mkdir -p $OUTDIR
echo "Outdir $OUTDIR"

lc_sizes=12
# lc_sizes=4
# EPOCH=400
EPOCH=500

# # PS-HPO
# SOURCE=$1
# DEVICE=$2
# # SOURCE=ccle
# # SOURCE=gdsc
# # SOURCE=ctrp
# # export CUDA_VISIBLE_DEVICES=1
# export CUDA_VISIBLE_DEVICES=$2
# echo "Processing source: $SOURCE"
# echo "CUDA device: $CUDA_VISIBLE_DEVICES"
# python src/main_lc.py \
#     -dp data/ml.dfs/data.$SOURCE.dd.ge.raw/data.$SOURCE.dd.ge.raw.parquet \
#     -sd data/ml.dfs/data.$SOURCE.dd.ge.raw/data.$SOURCE.dd.ge.raw.splits \
#     --split_id 0 \
#     --fea_prfx ge dd --fea_sep _ \
#     -t AUC -sc stnd --ml $MODEL \
#     --batch_size 64 --epoch $EPOCH \
#     --lc_sizes $lc_sizes \
#     --gout $OUTDIR/lc_${SOURCE}_ps_hpo \
#     --ps_hpo_dir k-tuner/${SOURCE}_tuner_out/ps_hpo

# # LS-HPO
# SOURCE=$1
# DEVICE=$2
# export CUDA_VISIBLE_DEVICES=$2
# echo "Processing source: $SOURCE"
# echo "CUDA device: $CUDA_VISIBLE_DEVICES"
# python src/main_lc.py \
#     -dp data/ml.dfs/data.$SOURCE.dd.ge.raw/data.$SOURCE.dd.ge.raw.parquet \
#     -sd data/ml.dfs/data.$SOURCE.dd.ge.raw/data.$SOURCE.dd.ge.raw.splits \
#     --split_id 0 \
#     --fea_prfx ge dd --fea_sep _ \
#     -t AUC -sc stnd --ml $MODEL \
#     --batch_size 64 --epoch $EPOCH \
#     --lc_sizes $lc_sizes \
#     --gout $OUTDIR/lc_${SOURCE}_ls_hpo \
#     --ls_hpo_dir k-tuner/${SOURCE}_tuner_out/ls_hpo

# Default HPs
SOURCE=$1
DEVICE=$2
MODEL=$3
export CUDA_VISIBLE_DEVICES=$2
echo "Processing source: $SOURCE"
echo "CUDA device: $CUDA_VISIBLE_DEVICES"
python src/main_lc.py \
    -dp data/ml.dfs/data.$SOURCE.dd.ge.raw/data.$SOURCE.dd.ge.raw.parquet \
    -sd data/ml.dfs/data.$SOURCE.dd.ge.raw/data.$SOURCE.dd.ge.raw.splits \
    --split_id 0 \
    --fea_prfx ge dd --fea_sep _ \
    -t AUC -sc stnd --ml $MODEL \
    --batch_size 64 --epoch $EPOCH \
    --batchnorm \
    --gout $OUTDIR/lc_${SOURCE}_${MODEL}_default \
    --lc_sizes $lc_sizes \

    # --lc_sizes_arr 88416 \
