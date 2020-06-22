#!/bin/bash

# The input that is assigned to "maindir" is the main output
# dir where keras-tuner dumps the results. For example:
# bash prep_hps_sets.bash k-tuner/gdsc_nn0_tuner_out

maindir=$1
cd $maindir
maindir=$(pwd)
echo "Main dir $maindir"

outdir=$maindir/ps_hpo
echo "Out dir $outdir"
mkdir -p $outdir

for fname in *; do
    if [[ "$fname" == *"tr_sz"* ]]; then
        echo ${fname}
        outpath=$outdir/$fname
        mkdir -p $outpath
        cp -r ${fname}/my_logs/* $outpath
    fi
done
