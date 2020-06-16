#!/bin/bash
# bsub -Is -W 0:30 -nnodes 1 -P MED106 $SHELL
# module load ibm-wml-ce/1.7.0-2
# conda activate gen

# Test
# CUDA_VISIBLE_DEVICES=1 
# python src/main_lc.py -dp $DATAPATH --lc_sizes_arr 128 2174 36937 --epoch $EPOCH --gout $gout --rout run"$split" -sc stnd > "$gout"/run"$split".log 2>&1
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_brk.sh 1 7 inter-parts/2 2 exec >inter_jsrun.log 2>&1 &

# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_brk.sh 1 0 gdsc-dbg data/ml.dfs/data.gdsc.dd.ge.raw/data.gdsc.dd.ge.raw.parquet data/ml.dfs/data.gdsc.dd.ge.raw/data.gdsc.dd.ge.raw.splits nn_reg0 exec >gdsc-dbg.log 2>&1 &
# GDSC:  [ 128 231 420 761 1379 2499 4528 8204  14865 26933 48799  88417]
# CTRP:  [ 128 253 500 988 1954 3862 7635 15093 29835 58976 116579 230442]

device=$1
split=$2
gout=$3
# SET=$4
dpath=$4
spath=$5
model=$6

echo "device $device"
echo "split: $split"
echo "gout:  $gout"
echo "dpath: $dpath"
echo "spath: $spath"
echo "model: $model"

gout="/gpfs/alpine/med106/scratch/$USER/$gout"
mkdir -p $gout
export CUDA_VISIBLE_DEVICES=$device

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Global outdir: $gout"

# lc_sizes=12
lc_sizes=8
# EPOCH=2
EPOCH=500

python src/main_lc.py \
    -dp $dpath \
    -sd $spath \
    --split_id $split \
    -t AUC -sc stnd --ml $model \
    --batch_size 64 --epoch $EPOCH \
    --batchnorm \
    --lc_sizes $lc_sizes \
    --gout $gout > "$gout"/run"$split".log 2>&1

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


