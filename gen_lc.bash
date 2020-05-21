#!/bin/bash

# SOURCE=gdsc
# SOURCE=ctrp
# SOURCE=ccle
SOURCE=$1
lc_sizes=12


python src/batch_lc.py -dp data/ml.data/data.$SOURCE.dsc.rna.raw/data.$SOURCE.dsc.rna.raw.parquet \
    -sd data/ml.data/data.$SOURCE.dsc.rna.raw/data.$SOURCE.dsc.rna.raw.splits -ns 100 --par_jobs 64 \
    --gout out/lc_$SOURCE -t AUC --lc_sizes $lc_sizes

