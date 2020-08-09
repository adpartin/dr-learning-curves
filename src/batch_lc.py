"""
A batch prcoessing that calls main_lc.py with the same set of parameters
but different split ids. This code uses the joblib Parallel.
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from glob import glob
from time import time

import numpy as np
from joblib import Parallel, delayed

filepath = Path(__file__).resolve().parent
import main_lc
from datasplit.split_getter import get_unq_split_ids


# Main func designed primarily for joblib Parallel
def run_split_ids(split_id, rout, *args):
    """ Use pre-computed split ids. """
    print('Calling run_split_ids ...')
    main_lc.main([ '--split_id', str(split_id), '--rout', 'run_'+str(rout), *args ])


# Main func designed primarily for joblib Parallel
def run_split_fly(n_splits, rout, *args):
    """ Generate split on the fly. """
    print('Calling run_split_fly ...')
    main_lc.main([ '--n_splits', str(n_splits), '--rout', 'run_'+str(rout), *args ])


parser = argparse.ArgumentParser()
parser.add_argument('-sd', '--splitdir',
                    required=False,
                    default=None,
                    type=str,
                    help='Full path to data splits (default: None).')
parser.add_argument('-ns', '--n_splits',
                    default=10,
                    type=int,
                    help='Use a subset of splits (default: 10).')
parser.add_argument('--par_jobs',
                    default=1,
                    type=int,
                    help='Number of joblib parallel jobs (default: 1).')
args, other_args = parser.parse_known_args()
print(args)


# Number of parallel jobs
par_jobs = int(args.par_jobs)
assert par_jobs > 0, f"The arg 'par_jobs' must be at least 1 (got {par_jobs})"

if args.splitdir is not None:
    main_fn = run_split_ids

    # 'splitdir' is also required for the function main_lc()
    other_args.extend(['--splitdir', args.splitdir])

    # Data file names
    split_pattern = '1fold_s*_*_id.csv'
    splitdir = Path(args.splitdir).resolve()
    all_split_files = glob( str(Path(splitdir, split_pattern)) )
    unq_split_ids = get_unq_split_ids( all_split_files )
    splits_arr = unq_split_ids
    runs_arr = splits_arr

    # Determine n_splits
    n_splits = np.min([ len(unq_split_ids), args.n_splits ])
else:
    main_fn = run_split_fly
    # splits_arr = np.arange( args.n_splits )
    splits_arr = [1 for _ in range(args.n_splits)]
    runs_arr = np.arange(args.n_splits)
    n_splits = args.n_splits
    # other_args.extend(['--splitdir', args.splitdir]) 

# Main execution
t0 = time()
if par_jobs > 1:
    # https://joblib.readthedocs.io/en/latest/parallel.html
    # results = Parallel(n_jobs=par_jobs, verbose=1)(
    #         delayed(run_split_ids)(split_id, *other_args) for split_id in unq_split_ids[:n_splits] )
    results = Parallel(n_jobs=par_jobs, verbose=1)(
            delayed(main_fn)(s, r, *other_args) for s, r in zip(splits_arr[:n_splits], runs_arr[:n_splits]) )
else:
    # for i, split_id in enumerate(unq_split_ids[:n_splits]):
    # for i, s in enumerate( splits_arr[:n_splits] ):
    for s, r in zip( splits_arr[:n_splits], runs_arr[:n_splits] ):
        print(f'Processing split {s}')
        # run_split_ids(split_id, *other_args)
        other_args_run = other_args.copy()
        # other_args_run.extend(['--rout', f'run_{s}']) 
        main_fn(s, r, *other_args_run) # only one split for every run

t_end = time() - t0
print('Runtime {:.1f} mins'.format(t_end/3600))
print('Done.')


