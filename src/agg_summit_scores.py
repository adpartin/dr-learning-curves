import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from glob import glob

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import matthews_corrcoef

filepath = Path(__file__).resolve().parent


def parse_args(args):
    parser = argparse.ArgumentParser(description='Aggregate LC data.')
    parser.add_argument('-r', '--res_dir',
                        required=True,
                        default=None,
                        type=str,
                        help='Global dir where LC data located (default: None).')
    args = parser.parse_args(args)
    return args


def agg_scores(cnt_dirs):
    """ ... """
    if len(cnt_dirs) == 0:
        print('Empty dir (no files).')
        return pd.DataFrame(), []

    dfs = []
    missing = []  # runs that were not completed (no scores.csv)

    for i, cnt_dr in enumerate(cnt_dirs):
        print(cnt_dr)
        # scores_path = glob( str(Path(d)/'**'/'lc_scores.csv') )
        sz_dirs = sorted(cnt_dr.glob('*/split0_sz*'))

        for j, sz_dr in enumerate(sz_dirs):
            scores_path = Path(sz_dr, 'scores.csv')  # Load scores

            if not scores_path.exists():
                dct = {'exp': str(sz_dr).split(os.sep)[-4],
                       'run': str(sz_dr).split(os.sep)[-3],
                       'sz': int(str(sz_dr).split(os.sep)[-1].split('_sz')[-1])
                       }
                missing.append(dct)

            else:
                df = pd.read_csv(scores_path)
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

                df['cnt']   = int(str(scores_path).split(os.sep)[-4].split('_')[0].split('cnt')[1])
                df['split'] = int(str(scores_path).split(os.sep)[-4].split('_')[1].split('split')[1])
                df['trial'] = int(str(scores_path).split(os.sep)[-4].split('_')[2].split('trial')[1])
                df['exp'] = str(scores_path).split(os.sep)[-5]
                dfs.append(df)
                del df

    scores = pd.concat(dfs, axis=0)
    scores = scores.sort_values(['split', 'trial']).reset_index(drop=True)

    missing = pd.DataFrame(missing)
    return scores, missing


def run(args):
    res_dir = Path(args.res_dir).resolve()

    cnt_dirs = sorted(res_dir.glob('cnt*split*trial*'))
    scores, missing = agg_scores(cnt_dirs)

    print(scores.shape)
    print('Unique splits: {}'.format(scores['split'].unique()))
    print('Unique sizes:  {}'.format(sorted(scores['tr_size'].unique())))

    scores.to_csv(res_dir/'all_scores.csv', index=False)
    if len(missing):
        missing.to_csv(res_dir/'missing.csv', index=False)
    print('Done.')


def main(args):
    args = parse_args(args)
    score = run(args)
    return None


if __name__ == '__main__':
    main(sys.argv[1:])
