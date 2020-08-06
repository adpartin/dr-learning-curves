#!/bin/bash

# Example:
# bash scripts/lc_lgb.bash ctrp 16

SOURCE=$1
PAR_JOBS=$2

MODEL="lgb"

OUTDIR=lc.out.${SOURCE}.lgb
mkdir -p $OUTDIR
echo "Outdir $OUTDIR"

# LC_SIZES=5
# LC_SIZES=7
# LC_SIZES=12
LC_SIZES=25

echo "Source: $SOURCE"
echo "Joblib jobs: $PAR_JOBS"

data_version=July2020
dpath=data/ml.dfs/$data_version/data.$SOURCE.dd.ge/data.$SOURCE.dd.ge.parquet 
spath=data/ml.dfs/$data_version/data.$SOURCE.dd.ge/data.$SOURCE.dd.ge.splits 
ls_hpo_dir=lgb.hpo.prms/${SOURCE}.lgb.hpo

# python src/main_lc.py \
#     -dp $dpath \
#     -sd $spath \
#     --ml $MODEL \
#     --gout $OUTDIR/lc.${SOURCE}.${MODEL}.ls_hpo \
#     --ls_hpo_dir $ls_hpo_dir \
#     --lc_sizes $LC_SIZES \
#     --n_jobs $PAR_JOBS 

python src/batch_lc.py \
    -dp $dpath \
    -sd $spath \
    --ml $MODEL \
    --gout $OUTDIR/lc.${SOURCE}.${MODEL}.ls_hpo \
    --ls_hpo_dir $ls_hpo_dir \
    --lc_sizes $LC_SIZES \
    --min_size 1024 \
    --n_jobs 8 \
    --n_splits 20 \
    --par_jobs $PAR_JOBS 
