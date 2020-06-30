#!/bin/bash
#BSUB -P med110
#BSUB -W 00:05
#BSUB -nnodes 2
#BSUB -J dr-crv-bsub
# ----------------------------------------------

# Before running bsub, load the required module and activate conda env!
# module load ibm-wml-ce/1.7.0-2
# conda activate gen

# bash bsub.sh > bsub.log
# echo "Bash version ${BASH_VERSION}"

# Number of splits
END_SPLIT=3
# END_SPLIT=19
START_SPLIT=0

cnt=0
run_trn () {
    src=$1
    model=$2
    log_dir=log.${src}.${model}
    mkdir -p $log_dir

    lc_sizes_arr=$3
    echo $lc_sizes_arr

    for split in $(seq $START_SPLIT 1 $END_SPLIT); do

        n_trial=3
        for trial in $(seq 1 $n_trial); do
            device=$(($cnt % 6))
            echo "Cnt $cnt; Split $split; Trial $trial; Device $device"

            # TODO: something is wrong with the "\"
            jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh $device $split $trial $src $model \
                "${lc_sizes_arr[@]}" $cnt exec >${log_dir}/cnt${cnt}_split${split}_trial${trial}.log 2>&1 &

            # jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh $device $split $trial $src $model "${lc_sizes_arr[@]}" $cnt exec >${log_dir}/cnt${cnt}_split${split}_trial${trial}.log 2>&1 &

            sleep 2
            cnt=$(($cnt + 1))
        done
    done
}

echo "Initial runs counter: ${cnt}"

# ================
#   GDSC
# lc_sizes_arr="2024 3798 7128 13377 25105 47113 88417"
# ================
lc_sizes_arr="2024 3798 7128 13377 25105 47113"
run_trn gdsc nn_reg0 "${lc_sizes_arr[@]}"
# run_trn gdsc nn_reg0 "${lc_sizes_arr}"

# lc_sizes_arr="2024 3798 7128 13377 25105 47113"
# run_trn gdsc nn_reg1 "${lc_sizes_arr[@]}"

# # lc_sizes_arr="88417"
# # run_trn gdsc nn_reg0 "${lc_sizes_arr[@]}"

# # lc_sizes_arr="88417"
# # run_trn gdsc nn_reg1 "${lc_sizes_arr[@]}"

# # ================
# #   CTRP
# # lc_sizes_arr="2024 4455 9809 21596 47545 104673 230442"
# # ================
# lc_sizes_arr="2024 4455 9809 21596 47545 104673"
# run_trn ctrp nn_reg0 "${lc_sizes_arr[@]}"

# lc_sizes_arr="2024 4455 9809 21596 47545 104673"
# run_trn ctrp nn_reg1 "${lc_sizes_arr[@]}"

# # lc_sizes_arr="230442"
# # run_trn ctrp nn_reg0 "${lc_sizes_arr[@]}"

# # lc_sizes_arr="230442"
# # run_trn ctrp nn_reg1 "${lc_sizes_arr[@]}"

echo "Final runs counter: ${cnt}"
# ========================================================================

# # ----------------------------------------
# #   GDSC nn_reg0
# # ----------------------------------------
# SOURCE=gdsc
# MODEL=nn_reg0
# log_dir=log.${SOURCE}.${MODEL}
# mkdir -p $log_dir
# # lc_sizes_arr="2024 3798 7128 13377 25105 47113 88417"

# cnt=0
# for split in $(seq $START_SPLIT 1 $END_SPLIT); do

#     n_trial=3
#     for trial in $(seq 1 $n_trial); do
#         device=$(($cnt % 6))
#         echo "Split $split; Trial $trial; Device $device"
#         jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh $device $split $trial $SOURCE $MODEL exec >${log_dir}/run"$split".log 2>&1 &
#         cnt=$(($cnt + 1))
#     done

# done

# # ----------------------------------------
# #   CTRP nn_reg0
# # ----------------------------------------
# SOURCE=ctrp
# MODEL=nn_reg0
# log_dir=log.${SOURCE}.${MODEL}
# mkdir -p $log_dir
# # [  2024   4455   9809  21596  47545 104673 230442]

# cnt=0
# for split in $(seq $START_SPLIT 1 $END_SPLIT); do

#     n_trial=3
#     for trial in $(seq 1 $n_trial); do
#         device=$(($cnt % 6))
#         echo "Split $split; Trial $trial; Device $device"
#         jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh $device $split $trial $SOURCE $MODEL exec >${log_dir}/run"$split".log 2>&1 &
#         cnt=$(($cnt + 1))
#     done

# done

