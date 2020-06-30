#!/bin/bash
#BSUB -P med110
#BSUB -W 06:00
#BSUB -nnodes 80
#BSUB -J dr-crv-bsub
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

            jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh $device $split $trial $src $model \
                "${lc_sizes_arr[@]}" $cnt exec >${log_dir}/cnt${cnt}_split${split}_trial${trial}.log 2>&1 &

            sleep 2
            cnt=$(($cnt + 1))
        done
    done
}

echo "Initial runs counter: ${cnt}"

# ================
#   GDSC --> 4 sets
# lc_sizes_arr="2024 3798 7128 13377 25105 47113 88417"
# ================
lc_sizes_arr="2024 3798 7128 13377 25105 47113"
run_trn gdsc nn_reg0 "${lc_sizes_arr[@]}"
# run_trn gdsc nn_reg0 "${lc_sizes_arr}"

lc_sizes_arr="2024 3798 7128 13377 25105 47113"
run_trn gdsc nn_reg1 "${lc_sizes_arr[@]}"

lc_sizes_arr="88417"
run_trn gdsc nn_reg0 "${lc_sizes_arr[@]}"

lc_sizes_arr="88417"
run_trn gdsc nn_reg1 "${lc_sizes_arr[@]}"

# ================
#   CTRP --> 4 sets
# lc_sizes_arr="2024 4455 9809 21596 47545 104673 230442"
# ================
lc_sizes_arr="2024 4455 9809 21596 47545 104673"
run_trn ctrp nn_reg0 "${lc_sizes_arr[@]}"

lc_sizes_arr="2024 4455 9809 21596 47545 104673"
run_trn ctrp nn_reg1 "${lc_sizes_arr[@]}"

lc_sizes_arr="230442"
run_trn ctrp nn_reg0 "${lc_sizes_arr[@]}"

lc_sizes_arr="230442"
run_trn ctrp nn_reg1 "${lc_sizes_arr[@]}"

echo "Final runs counter: ${cnt}"
# ========================================================================

