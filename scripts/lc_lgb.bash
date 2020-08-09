#!/bin/bash

# Example:
# bash scripts/lc_lgb.bash ctrp 16 none
# bash scripts/lc_lgb.bash nci60 2 random

SOURCE=$1
PAR_JOBS=$2
SAMPLING=$3

MODEL="lgb"

OUTDIR=lc.out.${SOURCE}.lgb
mkdir -p $OUTDIR
echo "Outdir $OUTDIR"

# LC_SIZES=5
# LC_SIZES=7
# LC_SIZES=12
# LC_SIZES=25
# LC_SIZES=50

echo "Source:   $SOURCE"
echo "Joblib:   $PAR_JOBS"
echo "Sampling: $SAMPLING"

data_version=July2020
# dpath=data/ml.dfs/$data_version/data.$SOURCE.dd.ge/data.$SOURCE.dd.ge.parquet 
# spath=data/ml.dfs/$data_version/data.$SOURCE.dd.ge/data.$SOURCE.dd.ge.splits 
# ls_hpo_dir=lgb.hpo.prms/${SOURCE}.lgb.hpo

# Data and splits path
None_Var="none"

if [[ "$SAMPLING" == "$None_Var" ]]; then
    dpath=data/ml.dfs/$data_version/data.$SOURCE.dd.ge/data.$SOURCE.dd.ge.parquet
    spath=data/ml.dfs/$data_version/data.$SOURCE.dd.ge/data.$SOURCE.dd.ge.splits
    # trn_dir=$base_dir/lc.${src}.${model}.ls_hpo
else
    dpath=data/ml.dfs/$data_version/data.$SOURCE.dd.ge.$SAMPLING/data.$SOURCE.dd.ge.parquet
    spath=data/ml.dfs/$data_version/data.$SOURCE.dd.ge.$SAMPLING/data.$SOURCE.dd.ge.splits
    # trn_dir=$base_dir/lc.${SOURCE}.${MODEL}.${SAMPLING}.ls_hpo
fi
# ls_hpo_dir=k-tuner/${SOURCE}_${MODEL}_tuner_out/ls_hpo
ls_hpo_dir=lgb.hpo.prms/${SOURCE}.lgb.hpo

# gout=$OUTDIR/lc.${SOURCE}.${MODEL}.ls_hpo
gout=$OUTDIR/lc.${SOURCE}.${MODEL}.dflt.low_range

echo "dpath: $dpath"
echo "spath: $spath"
echo "gout:  $gout"

# python src/main_lc.py \
#     -dp $dpath \
#     -sd $spath \
#     --ml $MODEL \
#     --gout $OUTDIR/lc.${SOURCE}.${MODEL}.ls_hpo \
#     --ls_hpo_dir $ls_hpo_dir \
#     --lc_sizes $LC_SIZES \
#     --n_jobs $PAR_JOBS 

# max_size=1024
min_size=4
max_size=2000
LC_SIZES=25

python src/batch_lc.py \
    -dp $dpath \
    -sd $spath \
    --ml $MODEL \
    --gout $gout \
    --lc_sizes $LC_SIZES \
    --min_size $min_size \
    --max_size $max_size \
    --n_jobs 8 \
    --n_splits 20 \
    --par_jobs $PAR_JOBS 

    # --ls_hpo_dir $ls_hpo_dir \
    # --min_size $min_size \
