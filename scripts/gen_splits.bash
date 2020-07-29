#!/bin/bash

SOURCE=$1
data_version=July2020

python ../ml-data-splits/src/main_data_split.py \
    -dp data/ml.dfs/$data_version/data.$SOURCE.dd.ge/data.$SOURCE.dd.ge.parquet \
    --gout data/ml.dfs/$data_version/data.$SOURCE.dd.ge \
    --trg_name AUC_bin \
    -ns 20 \
    -cvm strat \
    --te_size 0.1
