import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from glob import glob

import sklearn
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

fpath = Path(os.getcwd())
print('Current path:', fpath)


def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate learning curves.')

    # Input data
    parser.add_argument('--res_dir', required=True, default=None, type=str,
                        help='Global dir where learning curve are located (default: None).')
    args, other_args = parser.parse_known_args(args)
    return args


def agg_scores(dirs, verbose=True):
    dfs = []
    missing = [] # runs that were not completed (no lc_scores.csv)

    if len(dirs) == 0:
        print('Empty list (no files).')
        return pd.DataFrame(), []
    
    for i, d in enumerate(dirs):
        # print(i, d)
        scores_path = glob( str(Path(d)/'**'/'lc_scores.csv') )
        if len(scores_path) > 0:
            scores_path = scores_path[0]
        else:
            missing.append(d)
            continue
        df = pd.read_csv( scores_path )

        df = df.drop(columns='split')
        rname = scores_path.split('/')[2]
        df['cnt'] = int(rname.split('_')[0].split('cnt')[1])
        df['split'] = int(rname.split('_')[1].split('split')[1])
        df['trial'] = int(rname.split('_')[2].split('trial')[1])
        df['run'] = scores_path.split('/')[1]

        dfs.append(df)
    
    if len(dfs) == 0:
        return pd.DataFrame(), []
    
    scores = pd.concat(dfs, axis=0)
    scores = scores.sort_values(['split', 'trial']).reset_index(drop=True)
    if verbose:
        print(scores.shape)
        print('Unique splits:  {}'.format( scores['split'].unique()) )
        print('Unique sizes:   {}'.format( sorted(scores['tr_size'].unique())) )
    return scores, missing


def run(args):
    res_dir = Path( args['res_dir'] ).resolve()
    dir_name = res_dir.name # .split('.')[1]

    # run_dirs = glob(str(res_dir/'run*'))
    # scores = agg_scores(run_dirs)
    
    run_dirs = glob(str(res_dir/'cnt*split*trial*'))
    scores, missing = agg_scores(run_dirs); 
    
    print(scores.shape)
    print('Unique splits:  {}'.format( scores['split'].unique()) )
    print('Unique sizes:   {}'.format( sorted(scores['tr_size'].unique())) )

    # outdir = Path(fpath, f'lc.{exp}.{method}')
    # os.makedirs(outdir, exist_ok=True)
    # scores.to_csv(outdir/'all_scores.csv', index=False)

    save = True
    if save:
        scores.to_csv(res_dir/'all_scores.csv', index=False)
        # te_scores.to_csv(res_dir/'te_scores.csv', index=False)
        # te_scores_mae.to_csv(res_dir/'te_scores_mae.csv', index=False)    
    

def main(args):
    args = parse_args(args)
    # args = vars(args)
    score = run(args)
    return None
    

if __name__ == '__main__':
    main(sys.argv[1:])