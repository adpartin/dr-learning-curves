#!/bin/bash

data_version=July2020
rawdir=data/raw/$data_version
outdir=$rawdir/../../ml.dfs/$data_version

rsp_path=$rawdir/combined_single_response_rescaled_agg
cell_path=$rawdir/lincs1000/combined_rnaseq_data_lincs1000

dropna_th=0.1
r2fit_th=0.5

# ------------------------------------------------------------
# sources=gdsc2
sources=('gdsc1' 'gdsc2' 'ctrp' 'ccle')

drug_path=$rawdir/drug_info/dd.mordred.with.nans

for SOURCE in ${sources[@]}; do
    python src/build_dfs_july2020.py \
        --src $SOURCE \
        --rsp_path $rsp_path \
        --cell_path $cell_path \
        --drug_path $drug_path \
        --dropna_th $dropna_th \
        --r2fit_th $r2fit_th \
        --gout $outdir
done

# ------------------------------------------------------------
drug_path=$rawdir/NCI60_drugs_52k_smiles/dd.mordred.with.nans

python src/build_dfs_july2020.py \
    --src nci60 \
    --rsp_path $rsp_path \
    --cell_path $cell_path \
    --drug_path $drug_path \
    --dropna_th $dropna_th \
    --r2fit_th $r2fit_th \
    --n_samples 500000 \
    --flatten \
    --gout $outdir
