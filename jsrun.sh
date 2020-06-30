#!/bin/bash
# bsub -Is -W 0:30 -nnodes 1 -P MED106 $SHELL
# bsub -Is -W 0:30 -nnodes 1 -P MED110 $SHELL
# module load ibm-wml-ce/1.7.0-2
# conda activate gen

# Test
# CUDA_VISIBLE_DEVICES=1 
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh 0 0 1 gdsc nn_reg0 "2000 4000 6000" 1 exec >jsrun.log 2>&1 &

printf "\nInside jsrun ..."
PROJ=med110

# Inputs
device=$1
split=$2
trial=$3
src=$4
model=$5
lc_sizes_arr=$6
cnt=$7

printf "\n"
echo "device: $device"
echo "split:  $split"
echo "trial:  $trial"
echo "source: $src"
echo "model:  $model"
echo "sizes:  $lc_sizes_arr"
echo "cnt:    $cnt"

# # CUDA device
# TODO: specifying CUDA_VISIBLE_DEVICES messes up the GPU allocation!
# export CUDA_VISIBLE_DEVICES=$device
# echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Data and splits path
dpath=data/ml.dfs/data.${src}.dd.ge.raw/data.${src}.dd.ge.raw.parquet
spath=data/ml.dfs/data.${src}.dd.ge.raw/data.${src}.dd.ge.raw.splits
ps_hpo_dir=k-tuner/${src}_${model}_tuner_out/ps_hpo

printf "\n"
echo "dpath:  $dpath"
echo "spath:  $spath"
echo "ps-hpo: $ps_hpo_dir"

# Global outdir
base_dir=/gpfs/alpine/$PROJ/scratch/$USER
trn_dir=$base_dir/lc.${src}.${model}.ps_hpo
gout=$trn_dir/cnt${cnt}_split${split}_trial${trial}
mkdir -p $trn_dir
mkdir -p $gout
echo "gout: $gout"

# Train settings
# LC_SIZES=7
# EPOCH=1
EPOCH=3
# EPOCH=500

sleep 2

# TODO: why it doesn't work when I put "&" at the end??
python src/main_lc.py \
    -dp $dpath \
    -sd $spath \
    --split_id $split \
    --ml $model \
    --gout $gout \
    --epoch $EPOCH \
    --batchnorm \
    --ps_hpo_dir $ps_hpo_dir \
    --lc_sizes_arr $lc_sizes_arr > $trn_dir/cnt${cnt}.log 2>&1

    # --lc_sizes $LC_SIZES \
    # --min_size 2024 > "$gout"/run"$split".log 2>&1

sleep 2


