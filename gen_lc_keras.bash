#!/bin/bash

OUTDIR=lc.out
mkdir -p $OUTDIR
echo "Outdir $OUTDIR"

lc_sizes=12

SOURCE=$1
DEVICE=$2
# SOURCE=ccle
# SOURCE=gdsc
# SOURCE=ctrp
# export CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=$2
echo "Processing source: $SOURCE"
echo "CUDA device: $CUDA_VISIBLE_DEVICES"
python src/main_lc.py -dp data/ml.dfs/data.$SOURCE.dsc.rna.raw/data.$SOURCE.dsc.rna.raw.parquet \
    --fea_prfx ge dd --fea_sep _ \
    --gout $OUTDIR/lc_$SOURCE -t AUC --ml keras -sc stnd \
    --batch_size 64 --epoch 300 \
    --lc_sizes $lc_sizes --hp_dir k-tuner/${SOURCE}_tuner_out/lc_hp_sets 

# SOURCE=gdsc
# export CUDA_VISIBLE_DEVICES=1
# echo "Processing source: $SOURCE"
# echo "CUDA device: $CUDA_VISIBLE_DEVICES"
# python src/main_lc.py -dp data/ml.data/data.$SOURCE.dsc.rna.raw/data.$SOURCE.dsc.rna.raw.parquet \
#     -sd data/ml.data/data.$SOURCE.dsc.rna.raw/data.$SOURCE.dsc.rna.raw.splits \
#     --gout $OUTDIR/lc_$SOURCE -t AUC --ml keras -sc stnd \
#     --lr 0.0001 --batch_size 64 --epoch 500 \
#     --lc_sizes $lc_sizes 
#     # --lc_sizes_arr 20000 50000 70000 \
#     # > out_tmp/$SOURCE.log # 2>&1 &

# SOURCE=ctrp
# export CUDA_VISIBLE_DEVICES=2
# echo "Processing source: $SOURCE"
# echo "CUDA device: $CUDA_VISIBLE_DEVICES"
# python src/main_lc.py -dp data/ml.data/data.$SOURCE.dsc.rna.raw/data.$SOURCE.dsc.rna.raw.parquet \
#     -sd data/ml.data/data.$SOURCE.dsc.rna.raw/data.$SOURCE.dsc.rna.raw.splits \
#     --gout $OUTDIR/lc_$SOURCE -t AUC --ml keras -sc stnd \
#     --lr 0.0001 --epoch 500 \
#     --lc_sizes $lc_sizes 


