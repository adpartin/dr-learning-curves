#!/bin/bash

SOURCE=$1
data_version=July2020

dpath=data/ml.dfs/$data_version/data.$SOURCE.dd.ge/data.$SOURCE.dd.ge.parquet
gout=data/ml.dfs/$data_version/data.$SOURCE.dd.ge

# sampling=random
# sampling=flatten

# dpath=data/ml.dfs/$data_version/data.$SOURCE.dd.ge.$sampling/data.$SOURCE.dd.ge.parquet
# gout=data/ml.dfs/$data_version/data.$SOURCE.dd.ge.$sampling

python ../ml-data-splits/src/main_data_split.py \
    -dp $dpath \
    --gout $gout \
    --trg_name AUC_bin \
    -ns 20 \
    -cvm strat \
    --te_size 0.05
