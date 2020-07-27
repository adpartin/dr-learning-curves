#!/bin/bash

# SOURCE=gdsc
# SOURCE=ctrp
sources=('gdsc' 'ctrp' 'ccle')

for SOURCE in ${sources[@]}; do
    python src/build_dfs.py --src $SOURCE
done

