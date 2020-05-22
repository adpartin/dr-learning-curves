#!/bin/bash

OUTDIR=tmp_out
mkdir OUTDIR
echo "Outdir $OUTDIR"

# SOURCE=$1
# PAR_JOBS=$2
lc_sizes=12

# echo "Processing source: $SOURCE"
# echo "Parallel jobs: $PAR_JOBS"

SOURCE=gdsc
export CUDA_VISIBLE_DEVICES=1
echo "Processing source: $SOURCE"
echo "CUDA device: $CUDA_VISIBLE_DEVICES"
python src/main_lc.py -dp data/ml.data/data.$SOURCE.dsc.rna.raw/data.$SOURCE.dsc.rna.raw.parquet \
    -sd data/ml.data/data.$SOURCE.dsc.rna.raw/data.$SOURCE.dsc.rna.raw.splits \
    --gout $OUTDIR/lc_$SOURCE -t AUC --ml keras -sc stnd \
    --lr 0.0001 --batch_size 64 --epoch 500 \
    --lc_sizes $lc_sizes 
    # --lc_sizes_arr 20000 50000 70000 \
    # > out_tmp/$SOURCE.log # 2>&1 &

# SOURCE=ctrp
# export CUDA_VISIBLE_DEVICES=2
# echo "Processing source: $SOURCE"
# echo "CUDA device: $CUDA_VISIBLE_DEVICES"
# python src/main_lc.py -dp data/ml.data/data.$SOURCE.dsc.rna.raw/data.$SOURCE.dsc.rna.raw.parquet \
#     -sd data/ml.data/data.$SOURCE.dsc.rna.raw/data.$SOURCE.dsc.rna.raw.splits \
#     --gout $OUTDIR/lc_$SOURCE -t AUC --ml keras -sc stnd \
#     --lr 0.0001 --epoch 500 \
#     --lc_sizes $lc_sizes 


