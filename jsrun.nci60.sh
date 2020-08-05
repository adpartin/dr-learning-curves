#!/bin/bash
# bsub -Is -W 0:30 -nnodes 1 -P MED106 $SHELL
# bsub -Is -W 0:30 -nnodes 1 -P MED110 $SHELL
# module load ibm-wml-ce/1.7.0-2
# conda activate gen

# Test
# CUDA_VISIBLE_DEVICES=1 
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.nci60.sh 0 0 1 ctrp  nn_reg0 "2000 4000 6000" 1 none exec >jsrun.log 2>&1 &
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.nci60.sh 0 0 1 nci60 nn_reg0 "2000 4000 6000" 2 random exec >jsrun.log 2>&1 &
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.nci60.sh 0 0 1 nci60 nn_reg0 "2000 4000 6000" 3 flatten exec >jsrun.log 2>&1 &

printf "\nInside jsrun.nci60.sh"
PROJ=med110

# Inputs
device=$1
split=$2
trial=$3
src=$4
model=$5
lc_sizes_arr=$6
cnt=$7
sampling=$8

printf "\n"
echo "device: $device"
echo "split:  $split"
echo "trial:  $trial"
echo "source: $src"
echo "model:  $model"
echo "sizes:  $lc_sizes_arr"
echo "cnt:    $cnt"
echo "sampling: $sampling"

# CUDA device
# TODO: specifying CUDA_VISIBLE_DEVICES messes up the GPU allocation!
# export CUDA_VISIBLE_DEVICES=$device
# echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Global outdir
base_dir=/gpfs/alpine/$PROJ/scratch/$USER
# trn_dir=$base_dir/lc.${src}.${model}.ps_hpo
trn_dir=$base_dir/lc.${src}.${model}.ls_hpo
# gout=$trn_dir/cnt${cnt}_split${split}_trial${trial}
# mkdir -p $trn_dir
# mkdir -p $gout
# echo "gout: $gout"

# # Data and splits path
# dpath=data/ml.dfs/data.${src}.dd.ge/data.${src}.dd.ge.parquet
# spath=data/ml.dfs/data.${src}.dd.ge/data.${src}.dd.ge.splits
# # ps_hpo_dir=k-tuner/${src}_${model}_tuner_out/ps_hpo
# ls_hpo_dir=k-tuner/${src}_${model}_tuner_out/ls_hpo

# Data and splits path
None_Var="none"

if [[ "$sampling" == "$None_Var" ]]; then
    dpath=data/ml.dfs/July2020/data.${src}.dd.ge/data.${src}.dd.ge.parquet
    spath=data/ml.dfs/July2020/data.${src}.dd.ge/data.${src}.dd.ge.splits
    trn_dir=$base_dir/lc.${src}.${model}.ls_hpo
else
    dpath=data/ml.dfs/July2020/data.${src}.dd.ge.$sampling/data.${src}.dd.ge.parquet
    spath=data/ml.dfs/July2020/data.${src}.dd.ge.$sampling/data.${src}.dd.ge.splits
    trn_dir=$base_dir/lc.${src}.${model}.${sampling}.ls_hpo
fi
ls_hpo_dir=k-tuner/${src}_${model}_tuner_out/ls_hpo

printf "\n"
echo "dpath:  $dpath"
echo "spath:  $spath"
# echo "ps-hpo: $ps_hpo_dir"
echo "ls-hpo: $ls_hpo_dir"

gout=$trn_dir/cnt${cnt}_split${split}_trial${trial}
mkdir -p $trn_dir
mkdir -p $gout
echo "gout: $gout"

# # Global outdir
# base_dir=/gpfs/alpine/$PROJ/scratch/$USER
# # trn_dir=$base_dir/lc.${src}.${model}.ps_hpo
# trn_dir=$base_dir/lc.${src}.${model}.ls_hpo
# gout=$trn_dir/cnt${cnt}_split${split}_trial${trial}
# mkdir -p $trn_dir
# mkdir -p $gout
# echo "gout: $gout"

# Train settings
# LC_SIZES=7
# EPOCH=1
# EPOCH=2
EPOCH=500

sleep 1

# TODO: why it doesn't work when I put "&" at the end??
python src/main_lc.py \
    -dp $dpath \
    -sd $spath \
    --split_id $split \
    --ml $model \
    --gout $gout \
    --epoch $EPOCH \
    --batchnorm \
    --ls_hpo_dir $ls_hpo_dir \
    --lc_sizes_arr $lc_sizes_arr > $trn_dir/cnt${cnt}.log 2>&1

    # --ps_hpo_dir $ps_hpo_dir \

    # --lc_sizes $LC_SIZES \
    # --min_size 2024 > "$gout"/run"$split".log 2>&1

# sleep 1


