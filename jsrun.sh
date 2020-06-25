#!/bin/bash
# bsub -Is -W 0:30 -nnodes 1 -P MED106 $SHELL
# bsub -Is -W 0:30 -nnodes 1 -P MED110 $SHELL
# module load ibm-wml-ce/1.7.0-2
# conda activate gen

# Test
# CUDA_VISIBLE_DEVICES=1 
# python src/main_lc.py -dp $DATAPATH --lc_sizes_arr 128 2174 36937 --epoch $EPOCH --gout $gout --rout run"$split" -sc stnd > "$gout"/run"$split".log 2>&1
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_brk.sh 1 7 inter-parts/2 2 exec >inter_jsrun.log 2>&1 &

# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh 2 7 1 gdsc nn_reg0 data/ml.dfs/data.gdsc.dd.ge.raw/data.gdsc.dd.ge.raw.parquet data/ml.dfs/data.gdsc.dd.ge.raw/data.gdsc.dd.ge.raw.splits nn_reg0 exec >gdsc.log 2>&1 &

device=$1
split=$2
trial=$3
src=$4
model=$5
lc_sizes_arr=$6
# gout=$3
# SET=$4
# dpath=$4
# spath=$5

# CUDA device
export CUDA_VISIBLE_DEVICES=$device
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Data and splits path
dpath="data/ml.dfs/data.${src}.dd.ge.raw/data.${src}.dd.ge.raw.parquet"
spath="data/ml.dfs/data.${src}.dd.ge.raw/data.${src}.dd.ge.raw.splits"

# Global outdir
gout=lc.${src}.${model}.ps_hpo/trial_$trial
# gout=/gpfs/alpine/med106/scratch/$USER/$gout
gout=/gpfs/alpine/med110/scratch/$USER/$gout
mkdir -p $gout

# PS-HPO dir
ps_hpo_dir=k-tuner/${src}_${model}_tuner_out/ps_hpo

LC_SIZES=7
# LC_SIZES=12

EPOCH=1
# EPOCH=2
# EPOCH=500

echo "device: $device"
echo "split:  $split"
echo "trial:  $trial"
echo "source: $src"
echo "model:  $model"
echo "sizes:  $lc_sizes_arr"

echo "dpath:  $dpath"
echo "spath:  $spath"
echo "gout:   $gout"
echo "ps-hpo: $ps_hpo_dir"

python src/main_lc.py \
    -dp $dpath \
    -sd $spath \
    --split_id $split \
    --fea_prfx ge dd --fea_sep _ \
    -t AUC -sc stnd --ml $model \
    --batch_size 32 --epoch $EPOCH \
    --batchnorm \
    --gout $gout \
    --ps_hpo_dir $ps_hpo_dir \
    --lc_sizes_arr $lc_sizes_arr > "$gout"/run"$split".log 2>&1

    # --lc_sizes $LC_SIZES \
    # --min_size 2024 > "$gout"/run"$split".log 2>&1

# ---------------------------------------------------------
# # [128 329 845 2174 5589 14368 36937 94952 244088]
# #  1   2   9
# # [128 329 244088]
# set1="128 329 244088"
# #  3   4    8 
# # [845 2174 94952]
# set2="845 2174 94952"
# #  5    6     7 
# # [5589 14368 36937]
# set3="5589 14368 36937"

# if [[ $SET -eq 1 ]]; then
#     echo "LC subset set $SET"
#     python src/main_lc.py -dp $dpath -sd $spath --split_id $id --lc_sizes_arr $set1 \
#         --epoch $EPOCH --gout $gout --rout run"$id" -sc stnd --ml keras > "$gout"/run"$id".log 2>&1
# elif [[ $SET -eq 2 ]]; then
#     echo "LC subset set $SET"
#     python src/main_lc.py -dp $dpath -sd $spath --split_id $id --lc_sizes_arr $set2 \
#         --epoch $EPOCH --gout $gout --rout run"$id" -sc stnd --ml keras > "$gout"/run"$id".log 2>&1
# elif [[ $SET -eq 3 ]]; then
#     echo "LC subset set $SET"
#     python src/main_lc.py -dp $dpath -sd $spath --split_id $id --lc_sizes_arr $set3 \
#         --epoch $EPOCH --gout $gout --rout run"$id" -sc stnd --ml keras > "$gout"/run"$id".log 2>&1
# fi


