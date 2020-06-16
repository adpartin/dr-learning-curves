#!/bin/bash

OUTDIR=lc.out
mkdir -p $OUTDIR
echo "Outdir $OUTDIR"

lc_sizes=12

# SOURCE=gdsc
# SOURCE=ctrp
# SOURCE=ccle
SOURCE=$1
PAR_JOBS=$2
MODEL="lgb"

echo "Processing source: $SOURCE"
echo "Parallel jobs: $PAR_JOBS"

python src/batch_lc.py \
    -dp data/ml.dfs/data.$SOURCE.dd.ge.raw/data.$SOURCE.dd.ge.raw.parquet \
    -sd data/ml.dfs/data.$SOURCE.dd.ge.raw/data.$SOURCE.dd.ge.raw.splits \
    -ns 120 \
    --fea_prfx ge dd --fea_sep _ -t AUC \
    --ml $MODEL \
    --gout $OUTDIR/lc_${SOURCE}_${MODEL}_default \
    --lc_sizes $lc_sizes \
    --par_jobs $PAR_JOBS 

