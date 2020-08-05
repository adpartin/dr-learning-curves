#!/bin/bash
#BSUB -P med110
#BSUB -W 06:00
#BSUB -nnodes 60
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

            # sleep 1
            cnt=$(($cnt + 1))
        done
    done
}

echo "Initial runs counter: ${cnt}"

# ================
#   GDSC --> 4 sets
# lc_sizes_arr="2024 3798 7128 13377 25105 47113 88417"
# lc_sizes_arr=[128 189 281 417 619 919 1364 2024]
# lc_sizes_arr="128 189 281 417 619 919 1364"
# ================
# -----------
# Upper range
# -----------
# lc_sizes_arr="2024 3798 7128 13377 25105 47113"
# run_trn gdsc nn_reg0 "${lc_sizes_arr[@]}"

# lc_sizes_arr="2024 3798 7128 13377 25105 47113"
# run_trn gdsc nn_reg1 "${lc_sizes_arr[@]}"

# lc_sizes_arr="88417"
# run_trn gdsc nn_reg0 "${lc_sizes_arr[@]}"

# lc_sizes_arr="88417"
# run_trn gdsc nn_reg1 "${lc_sizes_arr[@]}"

# -----------
# Lower range
# -----------
# lc_sizes_arr="128 189 281 417 619 919 1364"
# run_trn gdsc nn_reg0 "${lc_sizes_arr[@]}"

# lc_sizes_arr="128 189 281 417 619 919 1364"
# run_trn gdsc nn_reg1 "${lc_sizes_arr[@]}"

# ================
#   CTRP --> 4 sets
# lc_sizes_arr="2024 4455 9809 21596 47545 104673 230442"
# lc_sizes_arr=[128 189 281 417 619 919 1364 2024]
# lc_sizes_arr="128 189 281 417 619 919 1364"
# ================
# -----------
# Upper range
# -----------
# lc_sizes_arr="2024 4455 9809 21596 47545 104673"
# run_trn ctrp nn_reg0 "${lc_sizes_arr[@]}"

# lc_sizes_arr="2024 4455 9809 21596 47545 104673"
# run_trn ctrp nn_reg1 "${lc_sizes_arr[@]}"

# lc_sizes_arr="230442"
# run_trn ctrp nn_reg0 "${lc_sizes_arr[@]}"

# lc_sizes_arr="230442"
# run_trn ctrp nn_reg1 "${lc_sizes_arr[@]}"

# -----------
# Lower range
# -----------
# lc_sizes_arr="128 189 281 417 619 919 1364"
# run_trn ctrp nn_reg0 "${lc_sizes_arr[@]}"

# lc_sizes_arr="128 189 281 417 619 919 1364"
# run_trn ctrp nn_reg1 "${lc_sizes_arr[@]}"

# ================
#   NCI60 --> 4 sets
# set= "2024 3919 7588 14694 28453 55095 106683 206575 400000"
# set1="2024 3919                                      400000"
# set2="          7588 14694                    206575       "
# set3="                     28453 55095 106683              "

# set= "2024 3272 5292 8557 13837 22375 36181 58506 94606 152979 247369 400000"
# set1="2024                      22375 36181                           400000"
# set2="     3272           13837             58506              247369       "
# set3="          5292 8557                         94606 152979              "
# ================
# -----------
# Upper range
# -----------
# set1="2024 22375 36181 400000"
# set2="3272 13837 58506 247369"
# set3="5292 8557  94606 152979"

# run_trn nci60 nn_reg1 "${set1[@]}"
# run_trn nci60 nn_reg1 "${set2[@]}"
# run_trn nci60 nn_reg1 "${set3[@]}"

# run_trn nci60 nn_reg0 "${set1[@]}"
# run_trn nci60 nn_reg0 "${set2[@]}"
# run_trn nci60 nn_reg0 "${set3[@]}"

# ================
#   NCI60 --> 4 sets
# set_A="5000 8658 14992 25962 44957 77850 134809 233442 404240 700000"
# set_B="453000 513000 580000 658000"
# ================
# -----------
# Upper range
# -----------
set1="700000 500000"
set2="650000 600000"
set3="400000 235000 135000 80000 45000 25000 15000 10000 5000 3000"

run_trn nci60 nn_reg1 "${set1[@]}"
run_trn nci60 nn_reg0 "${set1[@]}"

run_trn nci60 nn_reg1 "${set2[@]}"
run_trn nci60 nn_reg0 "${set2[@]}"

run_trn nci60 nn_reg1 "${set3[@]}"
run_trn nci60 nn_reg0 "${set3[@]}"

echo "Final runs counter: ${cnt}"
# ========================================================================

