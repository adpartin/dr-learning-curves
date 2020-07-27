#!/bin/bash

# SOURCE=gdsc
# SOURCE=ctrp
# SOURCE=ccle
SOURCE=$1

data_version=July2020

# # python ../covid-19/ml-data-splits/src/main_data_split.py \
# python ../ml-data-splits/src/main_data_split.py \
#     -dp data/ml.dfs/data.$SOURCE.dd.ge.raw/data.$SOURCE.dd.ge.raw.parquet \
#     --gout data/ml.dfs/data.$SOURCE.dd.ge.raw/ -ns 20 -cvm simple --te_size 0.1

# python ../covid-19/ml-data-splits/src/main_data_split.py \
python ../ml-data-splits/src/main_data_split.py \
    -dp data/ml.dfs/$data_version/data.$SOURCE.dd.ge/data.$SOURCE.dd.ge.parquet \
    --gout data/ml.dfs/$data_version/data.$SOURCE.dd.ge/ -ns 20 -cvm simple --te_size 0.1

