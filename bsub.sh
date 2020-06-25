#!/bin/bash
#BSUB -P med110
#BSUB -W 00:07
#BSUB -nnodes 1
#BSUB -J dr-crv-bsub
# ----------------------------------------------

# First load the required module and activate the conda env!
# module load ibm-wml-ce/1.7.0-2
# conda activate gen

echo "Bash version ${BASH_VERSION}"

# Number of splits
END_SPLIT=1
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
            echo "Split $split; Trial $trial; Device $device"
            jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh $device $split $trial $src $model "${lc_sizes_arr[@]}" exec >${log_dir}/split"$split"_trial"$trial".log 2>&1 &
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
# #   GDSC nn_reg1
# # ----------------------------------------
# SOURCE=gdsc
# MODEL=nn_reg1

# log_dir=log.${SOURCE}.${MODEL}
# mkdir -p $log_dir

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


# # ----------------------------------------
# #   CTRP nn_reg1
# # ----------------------------------------
# SOURCE=ctrp
# MODEL=nn_reg1

# log_dir=log.${SOURCE}.${MODEL}
# mkdir -p $log_dir

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
# # SETS=(1 2 3)

# # for SET in ${SETS[@]}; do
# #     out_dir=$GLOBAL_SUFX/set${SET}
# #     echo "Outdir $out_dir"
# #     for split in $(seq $START_SPLIT 1 $N_SPLITS); do
# #         device=$(($split % 6))
# #         echo "Set $SET; Split $split; Device $device"
# #         jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_brk.sh $device $split $out_dir $SET \
# #             exec >logs/run"$split".log 2>&1 &
# #     done
# # done

# ------------------------------------------------------------
# SET=1
# out_dir=$GLOBAL_SUFX/$SET
# echo "Dir $out_dir"
# for split in $(seq $START_SPLIT 1 $N_SPLITS); do
#     device=$(($split % 6))
#     echo "Set $SET; Split $split; Device $device"
#     jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_brk.sh $device $split $out_dir $SET exec >logs/run"$split".log 2>&1 &
# done

# SET=2
# out_dir=$GLOBAL_SUFX/$SET
# echo "Dir $out_dir"
# for split in $(seq $START_SPLIT 1 $N_SPLITS); do
#     device=$(($split % 6))
#     echo "Set $SET; Split $split; Device $device"
#     jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_brk.sh $device $split $out_dir $SET exec >logs/run"$split".log 2>&1 &
# done

# SET=3
# out_dir=$GLOBAL_SUFX/$SET
# echo "Dir $out_dir"
# for split in $(seq $START_SPLIT 1 $N_SPLITS); do
#     device=$(($split % 6))
#     echo "Set $SET; Split $split; Device $device"
#     jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_brk.sh $device $split $out_dir $SET exec >logs/run"$split".log 2>&1 &
# done
