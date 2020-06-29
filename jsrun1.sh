#!/bin/bash
# bsub -Is -W 0:30 -nnodes 1 -P MED110 $SHELL
# module load ibm-wml-ce/1.7.0-2
# conda activate gen

# Test
# CUDA_VISIBLE_DEVICES=1 
# jsrun -n 6 -a 1 -c 4 -g 1 ./jsrun1.sh gdsc nn_reg0 exec >jsrun.log 2>&1 &
# jsrun -n 3 -a 1 -c 4 -g 1 ./jsrun1.sh gdsc nn_reg0 exec >jsrun.log 2>&1 &
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun1.sh gdsc nn_reg0 "2000 5000 8000" exec >jsrun.log 2>&1 &

printf "\nInside jsrun ...\n"
src=$1
model=$2
lc_sizes_arr=$3

echo "source: $src"
echo "model:  $model"
echo "sizes:  $lc_sizes_arr"

# Data and splits path
dpath="data/ml.dfs/data.${src}.dd.ge.raw/data.${src}.dd.ge.raw.parquet"
spath="data/ml.dfs/data.${src}.dd.ge.raw/data.${src}.dd.ge.raw.splits"

# PS-HPO dir
ps_hpo_dir="k-tuner/${src}_${model}_tuner_out/ps_hpo"

echo "dpath:  $dpath"
echo "spath:  $spath"
echo "ps-hpo: $ps_hpo_dir"

EPOCH=1
# EPOCH=2
# EPOCH=500

# Number of splits
END_SPLIT=0
# END_SPLIT=1
# END_SPLIT=19
START_SPLIT=0

# cnt=0
cnt=-1
for split in $(seq $START_SPLIT 1 $END_SPLIT); do

    n_trial=3
    for trial in $(seq 1 $n_trial); do
        cnt=$(($cnt + 1)) # why this does not increment??

        device=$(($cnt % 6))
        export CUDA_VISIBLE_DEVICES=$device
        echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

        # Global outdir
        gout=lc.${src}.${model}.ps_hpo/cnt${cnt}_split${split}_trial${trial}
        gout=/gpfs/alpine/med110/scratch/$USER/$gout
        echo $gout
        mkdir -p $gout
        
        # TODO: it seems like the python script is not launched(??)
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
            --lc_sizes_arr $lc_sizes_arr > "$gout"/cnt"$cnt".log 2>&1 &
    done
done


