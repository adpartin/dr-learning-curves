#!/bin/bash

# SOURCE=gdsc
# SOURCE=ctrp
# SOURCE=ccle
SOURCE=$1
PAR_JOBS=$2
lc_sizes=12

echo "Processing source: $SOURCE"
echo "Parallel jobs: $PAR_JOBS"

python src/batch_lc.py -dp data/ml.data/data.$SOURCE.dsc.rna.raw/data.$SOURCE.dsc.rna.raw.parquet \
    -sd data/ml.data/data.$SOURCE.dsc.rna.raw/data.$SOURCE.dsc.rna.raw.splits \
    -ns 100 --gout out/lc_$SOURCE -t AUC --lc_sizes $lc_sizes --par_jobs $PAR_JOBS 

