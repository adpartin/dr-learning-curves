"""
Post-processing script.
Learning curve data are generated for different splits of a dataset.
This script aggregates LC results from different runs (i.e., different data splits).

Example:
python src/agg_results_from_runs.py --res_dir trn/ml.ADRP_pocket1_dock.lc/
python src/agg_results_from_runs.py --res_dir trn/ml.ADRP-ADPR_pocket1_dock.lc/
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.metrics import matthews_corrcoef

# Utils
from learningcurve.lc_plots import plot_lc_many_metric

# File path
filepath = Path(__file__).resolve().parent


def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate learning curves.')
    parser.add_argument('-r', '--res_dir',
                        required=True,
                        default=None,
                        type=str,
                        help='Global dir where LC data are located (default: None).')
    args, other_args = parser.parse_known_args(args)
    return args


def agg_scores(run_dirs):
    """ Aggregate scores from LC runs. Load scores from lc_scores.csv. """
    scores = []

    for i, r in enumerate(run_dirs):
        dpath = Path(r, 'lc_scores.csv')
        if not dpath.exists():
            continue

        scr = pd.read_csv(dpath)
        scr['split'] = Path(r).name
        scores.append(scr)

    scores = pd.concat(scores, axis=0)
    return scores


def agg_scores_02(run_dirs):
    """ Aggregate scores from LC runs. Instead of using lc_scores.csv, load
    scores from scores.csv that located in each split0_sz* folder. """
    dfs = []

    for rn_dr in run_dirs:
        print(rn_dr)

        sz_dirs = sorted(rn_dr.glob('*_sz*'))
        for sz_dr in sz_dirs:

            dpath = Path(sz_dr, 'scores.csv')
            if not dpath.exists():
                continue

            df = pd.read_csv(dpath)
            df = df.drop(columns='split')

            # --------------------
            # Add MCC to scores
            # --------------------
            preds_path = Path(sz_dr, 'preds_te.csv')  # load preds
            df_pred = pd.read_csv(preds_path)
            y_cls_true = df_pred['y_true'].map(lambda x: 0 if x > 0.5 else 1).values
            y_cls_pred = df_pred['y_pred'].map(lambda x: 0 if x > 0.5 else 1).values
            mcc = matthews_corrcoef(y_cls_true, y_cls_pred)
            df_mcc = pd.DataFrame([[df['tr_size'][0], 'te', 'mcc', mcc]],
                                  columns=['tr_size', 'set', 'metric', 'score'])
            df = pd.concat([df, df_mcc], axis=0).reset_index(drop=True)

            df['split'] = Path(rn_dr).name
            dfs.append(df)

    scores = pd.concat(dfs, axis=0)
    return scores


def run(args):
    res_dir = Path(args.res_dir).resolve()

    run_dirs = sorted(res_dir.glob('run*'))
    # scores = agg_scores(run_dirs)
    scores = agg_scores_02(run_dirs)

    print('Training set sizes:', np.unique(scores.tr_size))

    te_scores = scores[scores['set'] == 'te'].reset_index(drop=True)
    te_scores_mae = scores[(scores['metric'] == 'mean_absolute_error') &
                           (scores['set'] == 'te')].reset_index(drop=True)

    scores.to_csv(res_dir/'all_scores.csv', index=False)
    te_scores.to_csv(res_dir/'te_scores.csv', index=False)
    te_scores_mae.to_csv(res_dir/'te_scores_mae.csv', index=False)

    kwargs = {'tr_set': 'te', 'xtick_scale': 'log2', 'ytick_scale': 'log2'}
    ax = plot_lc_many_metric(scores, metrics=['mean_absolute_error', 'r2', 'mcc'],
                             outdir=res_dir, **kwargs)


def main(args):
    args = parse_args(args)
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
