#!/bin/bash

# SOURCE=gdsc
# SOURCE=ctrp
# SOURCE=ccle
SOURCE=$1

python ../covid-19/ml-data-splits/src/main_data_split.py \
    -dp data/ml.data/data.$SOURCE.dsc.rna.raw/data.$SOURCE.dsc.rna.raw.parquet \
    --gout data/ml.data/data.$SOURCE.dsc.rna.raw/ -ns 100 -cvm simple --te_size 0.1

