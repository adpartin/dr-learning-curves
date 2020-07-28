#!/bin/bash

# sources=gdsc
# sources=gdsc2
# SOURCE=ctrp
# sources=('gdsc' 'ctrp' 'ccle')
# sources=('gdsc2' 'ctrp' 'ccle')
sources=nci60

data_version=July2020
rawdir=data/raw/$data_version
outdir=$rawdir/../../ml.dfs/$data_version

# Response path
rsp_path=$rawdir/combined_single_response_rescaled_agg

# Cell path
cell_path=$rawdir/lincs1000/combined_rnaseq_data_lincs1000

# Drug path
# drug_path=$rawdir/drug_info/dd.mordred.with.nans
drug_path=$rawdir/NCI60_drugs_52k_smiles/dd.mordred.with.nans

for SOURCE in ${sources[@]}; do
    python src/build_dfs_july2020.py \
        --src $SOURCE \
        --rsp_path $rsp_path \
        --cell_path $cell_path \
        --drug_path $drug_path \
        --dropna_th 0.1 \
        --gout $outdir
done

