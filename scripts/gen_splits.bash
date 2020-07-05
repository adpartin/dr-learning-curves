#!/bin/bash

# SOURCE=gdsc
# SOURCE=ctrp
# SOURCE=ccle
SOURCE=$1

python ../covid-19/ml-data-splits/src/main_data_split.py \
    -dp data/ml.data/data.$SOURCE.dd.ge.raw/data.$SOURCE.dd.ge.raw.parquet \
    --gout data/ml.data/data.$SOURCE.dd.ge.raw/ -ns 20 -cvm simple --te_size 0.1

