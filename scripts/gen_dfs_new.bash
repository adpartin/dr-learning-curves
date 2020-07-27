#!/bin/bash

# sources=gdsc
# sources=gdsc2
# SOURCE=ctrp
# sources=('gdsc' 'ctrp' 'ccle')
sources=('gdsc2' 'ctrp' 'ccle')

data_version=July2020
rawdir=data/raw/$data_version
outdir=$rawdir/../../ml.dfs/$data_version

for SOURCE in ${sources[@]}; do
    python src/build_dfs_july2020.py \
        --src $SOURCE \
        --rsp_path $rawdir/combined_single_response_rescaled_agg \
        --drug_path $rawdir/drug_info/dd.mordred.with.nans \
        --cell_path $rawdir/lincs1000/combined_rnaseq_data_lincs1000 \
        --gout $outdir
done

