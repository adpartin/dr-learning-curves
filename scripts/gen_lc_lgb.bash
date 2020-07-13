#!/bin/bash

# Call this function from the main project dir!
OUTDIR=lc.out
mkdir -p $OUTDIR
echo "Outdir $OUTDIR"

# LC_SIZES=5
# LC_SIZES=7
LC_SIZES=12
LC_SIZES=25

# Default HPs
SOURCE=$1
PAR_JOBS=$2
MODEL="lgb"

echo "Processing source: $SOURCE"
echo "Parallel jobs: $PAR_JOBS"

dpath=data/ml.dfs/data.$SOURCE.dd.ge.raw/data.$SOURCE.dd.ge.raw.parquet 
spath=data/ml.dfs/data.$SOURCE.dd.ge.raw/data.$SOURCE.dd.ge.raw.splits 
# ps_hpo_dir=k-tuner/${SOURCE}_${MODEL}_tuner_out/ps_hpo
ls_hpo_dir=lgb.hpo/${SOURCE}.lgb.hpo


# python src/batch_lc.py \
python src/main_lc.py \
    -dp $dpath \
    -sd $spath \
    --ml $MODEL \
    --gout $OUTDIR/lc.${SOURCE}.${MODEL}.ls_hpo \
    --ls_hpo_dir $ls_hpo_dir \
    --lc_sizes $LC_SIZES \
    --n_jobs $PAR_JOBS 


    # --gout $OUTDIR/lc.${SOURCE}.${MODEL} \

    # --gout $OUTDIR/lc.${SOURCE}.${MODEL}.ls_hpo \
    # --ls_hpo_dir $ls_hpo_dir \

    # --gout $OUTDIR/lc_${SOURCE}_${MODEL} \
    # -ns 60 \
    # --par_jobs $PAR_JOBS 
