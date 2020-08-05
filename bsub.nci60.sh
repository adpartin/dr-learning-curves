#!/bin/bash
#BSUB -P med110
#BSUB -W 12:00
#BSUB -nnodes 100
#BSUB -J dr-crv-nci60
# ----------------------------------------------

# Before running bsub, load the required module and activate conda env!
# module load ibm-wml-ce/1.7.0-2
# conda activate gen

# bash bsub.sh > bsub.log
# echo "Bash version ${BASH_VERSION}"

# Determine the number of nodes:
# (num of splits) x (num of trials) x (num of sets) / (num of GPUs per node)
# 20 x 3 x 8 / 6 = 80

# Number of splits
# END_SPLIT=3
END_SPLIT=19
START_SPLIT=0

cnt=0
run_trn () {
    src=$1
    model=$2
    sampling=$4

    log_dir=log.${src}.${model}
    mkdir -p $log_dir

    printf "\n"
    echo "Source $src; Model $model"
    lc_sizes_arr=$3
    echo $lc_sizes_arr

    for split in $(seq $START_SPLIT 1 $END_SPLIT); do

        n_trial=3
        for trial in $(seq 1 $n_trial); do
            device=$(($cnt % 6))
            echo "Cnt $cnt; Split $split; Trial $trial; Device $device"

            jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.nci60.sh $device $split $trial $src $model \
                "${lc_sizes_arr[@]}" $cnt $sampling exec >${log_dir}/cnt${cnt}_split${split}_trial${trial}.log 2>&1 &

            # sleep 1
            cnt=$(($cnt + 1))
        done
    done
}

echo "Initial runs counter: ${cnt}"

# ================
#   NCI60
# set_A=[10000 13510 18252 24660 33316 45011 60811 82158 110998 149961 202602 273721 369806 499618 674999]
# set_B=[500000 521902 544763 568626 593534 619533 646672 674999]
# set1=[10000 13510 18252 24660 33316 45000 61000 82000 111000 150000 202000 273000 367000 500000 625000 674999]

# set_C=[500000 525644 552604 580947 610743 642067 674999]
# set2=[        525644 552604 580947 610743 642067       ]
# ================
# set1="674999 625000 499618 369806 273721 202602 149961 110998 82158 60811 45011 33316 24660 18252 13510 10000"
# set2="642067 610743 580947 552604 525644"
# Done:  674999 642067 {625000 610743}
# set1="              499618 369806 273721 202602 149961 110998 82158 60811 45011 33316 24660 18252 13510 10000"
# set2="              580947 552604 525644"
set1="              499618 369806 273721 202602 149961 110998  60811  33316  18252  10000"
set2="              570000 525644 410000"

# nci60_nn_reg0_random
# 369806    33
# 552604    32
# 580947    28
# 499618    25
# 273721     2
# done (set 1): 674999 625000 (499618 369806 273721)
# done (set 2): 642067 610743 (580947 552604)

# nci60_nn_reg1_random
# 499618    56
# 580947    54
# 610743     6
# 625000     4
# done (set 1): 674999 (625000 499618)
# done (set 2): 642067 (610743 580947 552604)

# nci60_nn_reg0_flatten
# 552604    49
# 369806    33
# 273721    18
# 525644     9
# 202602     5
# 499618     4
# 580947     2
# done (set 1): 674999 625000 (499618 369806 273721 202602)
# done (set 2): 642067 610743 (580947 552604 525644)

# nci60_nn_reg0_flatten
# 580947    56
# 499618    49
# 625000     9
# 610743     4
# 369806     2
# done (set 1): 674999 (625000 499618 369806)
# done (set 2): 642067 (610743 580947)

# Set 1
run_trn nci60 nn_reg1 "${set1[@]}" random
run_trn nci60 nn_reg1 "${set1[@]}" flatten
run_trn nci60 nn_reg0 "${set1[@]}" random
run_trn nci60 nn_reg0 "${set1[@]}" flatten

# Set 2
run_trn nci60 nn_reg1 "${set2[@]}" random
run_trn nci60 nn_reg1 "${set2[@]}" flatten
run_trn nci60 nn_reg0 "${set2[@]}" random
run_trn nci60 nn_reg0 "${set2[@]}" flatten

# ================
#   CTRP
# ================
# set1="10000 11360 12906 14661 16656 18922 21497 24422 27744 31519 35807 40679 46213 52500 59643 67757 76976 87448 99346 112862 128217 145661 165478 187991 213568"
# set1=213568, 187991, 165478, 145661, 128217, 112862,  99346,  87448, 76976,  67757,  59643,  52500,  46213,  40679,  35807,  31519, 27744,  24422,  21497,  18922,  16656,  14661,  12906,  11360, 10000

# nn_reg0
# 112862    27
# 99346     21
# 87448      9
# 128217     3
# not done: 87448  99346 112862 128217 145661 165478 187991 213568"

# nn_reg1
# 112862    30
# 99346     17
# 128217    13
# not done: 99346 112862 128217 145661 165478 187991 213568"

set2="213568 187991 165478 145661 128217 112862"

run_trn ctrp nn_reg1 "${set1[@]}" none
run_trn ctrp nn_reg0 "${set1[@]}" none

# ========================================================================
echo "Final runs counter: ${cnt}"
# ========================================================================
