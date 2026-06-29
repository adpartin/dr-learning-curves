#!/bin/bash

# Example:
# bash scripts/lc_lgb.bash ctrp 16 none
# bash scripts/lc_lgb.bash gdsc1 16 none
# bash scripts/lc_lgb.bash gdsc2 16 none
# bash scripts/lc_lgb.bash nci60 2 random

SOURCE=$1
PAR_JOBS=$2
SAMPLING=$3

MODEL="lgb"

OUTDIR=lc.out.${SOURCE}.lgb
# OUTDIR=lc.out.new.r2fit_03
mkdir -p $OUTDIR
echo "Outdir $OUTDIR"

# LC_SIZES=10
# LC_SIZES=40
LC_SIZES=50

echo "Source:   $SOURCE"
echo "Joblib:   $PAR_JOBS"
echo "Sampling: $SAMPLING"

data_version=July2020

# Data and splits path
None_Var="none"

if [[ "$SAMPLING" == "$None_Var" ]]; then
    dpath=data/ml.dfs/$data_version/data.$SOURCE.dd.ge/data.$SOURCE.dd.ge.parquet
    spath=data/ml.dfs/$data_version/data.$SOURCE.dd.ge/data.$SOURCE.dd.ge.splits
else
    dpath=data/ml.dfs/$data_version/data.$SOURCE.dd.ge.$SAMPLING/data.$SOURCE.dd.ge.parquet
    spath=data/ml.dfs/$data_version/data.$SOURCE.dd.ge.$SAMPLING/data.$SOURCE.dd.ge.splits
fi
# ls_hpo_dir=k-tuner/${SOURCE}_${MODEL}_tuner_out/ls_hpo
ls_hpo_dir=lgb.hpo.prms/${SOURCE}.lgb.hpo


echo "dpath: $dpath"
echo "spath: $spath"

min_size=10
# min_size=20000
# min_size=50000
# min_size=100000

# Default
gout=$OUTDIR/lc.${SOURCE}.${MODEL}.dflt
echo "gout:  $gout"
python src/batch_lc.py \
    -dp $dpath \
    -sd $spath \
    --ml $MODEL \
    --gout $gout \
    --lc_sizes $LC_SIZES \
    --min_size $min_size \
    --n_jobs 8 \
    --n_splits 20 \
    --par_jobs $PAR_JOBS 

# # HPO
# gout=$OUTDIR/lc.${SOURCE}.${MODEL}.ls_hpo
# echo "gout:  $gout"
# python src/batch_lc.py \
#     -dp $dpath \
#     -sd $spath \
#     --ml $MODEL \
#     --gout $gout \
#     --ls_hpo_dir $ls_hpo_dir \
#     --lc_sizes $LC_SIZES \
#     --min_size $min_size \
#     --n_jobs 8 \
#     --n_splits 20 \
#     --par_jobs $PAR_JOBS 
