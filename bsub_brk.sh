#!/bin/bash
#BSUB -P med106
#BSUB -W 06:00
#BSUB -nnodes 80
#BSUB -J dr-crv-bsub
# ----------------------------------------------

# First load the required module and activate the conda env!
# module load ibm-wml-ce/1.7.0-2
# conda activate gen

echo "Bash version ${BASH_VERSION}"

# Set resources based on number of splits
# END_SPLIT=11
END_SPLIT=119
START_SPLIT=0

# ----------------------------------------
#   GDSC nn_reg0
# ----------------------------------------
SOURCE=gdsc
MODEL=nn_reg0

GOUT="${SOURCE}.${MODEL}.lc.out"
DPATH="data/ml.dfs/data.${SOURCE}.dd.ge.raw/data.${SOURCE}.dd.ge.raw.parquet"
SPATH="data/ml.dfs/data.${SOURCE}.dd.ge.raw/data.${SOURCE}.dd.ge.raw.splits"
echo "GOUT $GOUT"

log_dir=logs.${SOURCE}.${MODEL}
mkdir -p $log_dir

for split in $(seq $START_SPLIT 1 $END_SPLIT); do
    device=$(($split % 6))
    echo "Split $split; Device $device"
    jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_brk.sh $device $split $GOUT $DPATH $SPATH $MODEL exec >${log_dir}/run"$split".log 2>&1 &
done

# ----------------------------------------
#   GDSC nn_reg1
# ----------------------------------------
SOURCE=gdsc
MODEL=nn_reg1

GOUT="${SOURCE}.${MODEL}.lc.out"
DPATH="data/ml.dfs/data.${SOURCE}.dd.ge.raw/data.${SOURCE}.dd.ge.raw.parquet"
SPATH="data/ml.dfs/data.${SOURCE}.dd.ge.raw/data.${SOURCE}.dd.ge.raw.splits"
echo "GOUT $GOUT"

log_dir=logs.${SOURCE}.${MODEL}
mkdir -p $log_dir

for split in $(seq $START_SPLIT 1 $END_SPLIT); do
    device=$(($split % 6))
    echo "Split $split; Device $device"
    jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_brk.sh $device $split $GOUT $DPATH $SPATH $MODEL exec >${log_dir}/run"$split".log 2>&1 &
done

# ----------------------------------------
#   CTRP nn_reg0
# ----------------------------------------
SOURCE=ctrp
MODEL=nn_reg0

GOUT="${SOURCE}.${MODEL}.lc.out"
DPATH="data/ml.dfs/data.${SOURCE}.dd.ge.raw/data.${SOURCE}.dd.ge.raw.parquet"
SPATH="data/ml.dfs/data.${SOURCE}.dd.ge.raw/data.${SOURCE}.dd.ge.raw.splits"
echo "GOUT $GOUT"

log_dir=logs.${SOURCE}.${MODEL}
mkdir -p $log_dir

for split in $(seq $START_SPLIT 1 $END_SPLIT); do
    device=$(($split % 6))
    echo "Split $split; Device $device"
    jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_brk.sh $device $split $GOUT $DPATH $SPATH $MODEL exec >${log_dir}/run"$split".log 2>&1 &
done

# ----------------------------------------
#   CTRP nn_reg1
# ----------------------------------------
SOURCE=ctrp
MODEL=nn_reg1

GOUT="${SOURCE}.${MODEL}.lc.out"
DPATH="data/ml.dfs/data.${SOURCE}.dd.ge.raw/data.${SOURCE}.dd.ge.raw.parquet"
SPATH="data/ml.dfs/data.${SOURCE}.dd.ge.raw/data.${SOURCE}.dd.ge.raw.splits"
echo "GOUT $GOUT"
mkdir -p logs

log_dir=logs.${SOURCE}.${MODEL}
mkdir -p $log_dir

for split in $(seq $START_SPLIT 1 $END_SPLIT); do
    device=$(($split % 6))
    echo "Split $split; Device $device"
    jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_brk.sh $device $split $GOUT $DPATH $SPATH $MODEL exec >${log_dir}/run"$split".log 2>&1 &
done


# ----------------------------------------
# SETS=(1 2 3)

# for SET in ${SETS[@]}; do
#     out_dir=$GLOBAL_SUFX/set${SET}
#     echo "Outdir $out_dir"
#     for split in $(seq $START_SPLIT 1 $N_SPLITS); do
#         device=$(($split % 6))
#         echo "Set $SET; Split $split; Device $device"
#         jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_brk.sh $device $split $out_dir $SET \
#             exec >logs/run"$split".log 2>&1 &
#     done
# done

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
