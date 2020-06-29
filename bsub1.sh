#!/bin/bash
#BSUB -P med110
#BSUB -W 00:07
#BSUB -nnodes 1
#BSUB -J dr-crv-bsub
# ----------------------------------------------

# Before running bsub, load the required module and activate conda env!
# module load ibm-wml-ce/1.7.0-2
# conda activate gen

echo "Bash version ${BASH_VERSION}"

run_trn1 () {
    src=$1
    model=$2
    printf "\nBsub train $src and $model ...\n"

    log_dir=log.${src}.${model}
    mkdir -p $log_dir

    lc_sizes_arr=$3
    echo $lc_sizes_arr

    jsrun -n 6 -a 1 -c 4 -g 1 ./jsrun1.sh $src $model \
        "${lc_sizes_arr[@]}" exec >${log_dir}/${src}.${model}.log 2>&1 &
}

lc_sizes_arr="2024 3798 7128 13377 25105 47113"
run_trn1 gdsc nn_reg0 "${lc_sizes_arr[@]}"
