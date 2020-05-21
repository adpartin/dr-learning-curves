from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from time import time
from pprint import pprint, pformat
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# File path
filepath = Path(__file__).resolve().parent

# Utils
from utils.classlogger import Logger
from utils.utils import load_data, dump_dict, get_print_func
from ml.scale import scale_fea
from ml.data import extract_subset_fea

import learningcurve as lc
from learningcurve.lrn_crv import LearningCurve
import learningcurve.lc_plots as lc_plots 
    

def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate learning curves (LCs).')
    # Input data
    parser.add_argument('-dp', '--datapath', required=True, default=None, type=str,
                        help='Full path to data (default: None).')
    # Pre-computed data splits
    parser.add_argument('-sd', '--splitdir', default=None, type=str,
                        help='Full path to data splits (default: None).')
    parser.add_argument('--split_id', default=0, type=int, help='Split id (default: 0).')
    # Number of data splits (generated on the fly)
    parser.add_argument('--n_splits', default=1, type=int,
                        help='Number of splits for which to generate LCs (computed automatically) .\
                        Used only if pre-computed splits not provided (default: 3).')
    # Out dirs
    parser.add_argument('--gout', default=None, type=str,
                        help='Gloabl outdir to dump all the results (i.e., splits/runs) (default: None).')
    parser.add_argument('--rout', default=None, type=str,
                        help='Run outdir specific run/split (default: None).')
    # Target to predict
    parser.add_argument('-t', '--trg_name', default='reg', type=str, choices=['AUC'],
                        help='Name of target variable (default: reg).')
    # Feature types
    parser.add_argument('-fp', '--fea_prfx', nargs='+', default=['DD','GE'], choices=['DD','GE'],
                        help='Prefix to identify the features (default: ...).')
    parser.add_argument('-fs', '--fea_sep', default='_', choices=['.','_'],
                        help="Separator btw fea prefix and fea name (default: '.').")
    # Feature scaling
    parser.add_argument('-sc', '--scaler', default=None, type=str, choices=['stnd', 'minmax', 'rbst'],
                        help='Feature normalization method (stnd, minmax, rbst) (default: None).')    
    # Learning curve
    parser.add_argument('--lc_step_scale', default='log', type=str, choices=['log', 'linear'],
                        help='Scale of progressive sampling of subset sizes in a learning curve (log2, log, log10, linear) (default: log).')
    parser.add_argument('--min_size', default=128, type=int, help='The lower bound for the subset size (default: 128).')
    parser.add_argument('--max_size', default=None, type=int, help='The upper bound for the subset size (default: None).')
    parser.add_argument('--lc_sizes', default=5, type=int, help='Number of subset sizes (default: 5).')
    parser.add_argument('--lc_sizes_arr', nargs='+', type=int, default=None, help='List of the actual sizes in the learning curve plot (default: None).')
    parser.add_argument('--save_model', action='store_true', help='Whether to trained models (default: False).')
    parser.add_argument('--plot_fit', action='store_true', help='Whether to generate the fit (default: False).')
    # HPs
    parser.add_argument('--ml', default='lgb', type=str, choices=['lgb', 'keras'], help='Choose ML model (default: lgb).')
    parser.add_argument('--epoch', default=1, type=int, help='Number of epochs (default: None).')
    parser.add_argument('--hp_file', default=None, type=str, help='File containing training hyperparameters (default: None).')
    parser.add_argument('--hpo_metric', default='mean_absolute_error', type=str, choices=['mean_absolute_error'],
                        help='Metric for HPO evaluation. Required for UPF workflow on Theta HPC (default: mean_absolute_error).')    
    # Other
    parser.add_argument('--n_jobs', default=8, type=int, help='Default: 8.')
    args, other_args = parser.parse_known_args(args)
    return args


def run(args):
    t0 = time()
    datapath = Path( args['datapath'] ).resolve()

    if args['splitdir'] is None:
        splitdir = None
    else:
        splitdir = Path( args['splitdir'] ).resolve()
    split_id = args['split_id']

    # -----------------------------------------------
    #       Global outdir
    # -----------------------------------------------
    if args['gout'] is not None:
        gout = Path( args['gout'] ).resolve()
    else:
        gout = filepath.parent/'trn'
        gout = gout/datapath.with_suffix('.lc').name
    args['gout'] = str(gout)
    os.makedirs(gout, exist_ok=True)
    
    # -----------------------------------------------
    #       Run (single split) outdir
    # -----------------------------------------------
    if args['rout'] is not None:
        rout = gout/args['rout']
    else:
        if splitdir is None:
            # rout = gout/f'run'
            rout = gout
        else:
            rout = gout/f'run_{split_id}'
    args['rout'] = str(rout)
    os.makedirs(rout, exist_ok=True)
    
    # -----------------------------------------------
    #       Logger
    # -----------------------------------------------
    lg = Logger( rout/'lc.log' )
    print_fn = get_print_func( lg.logger )
    print_fn(f'File path: {filepath}')
    print_fn(f'\n{pformat(args)}')
    dump_dict(args, outpath=rout/'trn.args.txt') # dump args.
    
    # -----------------------------------------------
    #       Load data
    # -----------------------------------------------
    print_fn('\nLoad master dataset.')
    data = load_data( datapath )
    print_fn('data.shape {}'.format(data.shape))
    
    # Get features (x), target (y), and meta
    fea_list = args['fea_prfx']
    fea_sep = args['fea_sep']
    xdata = extract_subset_fea(data, fea_list=fea_list, fea_sep=fea_sep)
    meta = data.drop( columns=xdata.columns )
    ydata = meta[[ args['trg_name'] ]]
    del data
    
    # -----------------------------------------------
    #       Scale features
    # -----------------------------------------------
    xdata = scale_fea(xdata=xdata, scaler_name=args['scaler'])    
    
    # -----------------------------------------------
    #       Data splits
    # -----------------------------------------------
    if splitdir is None:
        cv_lists = None
    else:
        split_pattern = f'1fold_s{split_id}_*_id.csv'
        single_split_files = glob( str(splitdir/split_pattern) )

        # Get indices for the split
        assert len(single_split_files) >= 2, f'The split {s} contains only one file.'
        for id_file in single_split_files:
            if 'tr_id' in id_file:
                tr_id = load_data( id_file ).values.reshape(-1,)
            elif 'vl_id' in id_file:
                vl_id = load_data( id_file ).values.reshape(-1,)
            elif 'te_id' in id_file:
                te_id = load_data( id_file ).values.reshape(-1,)

        cv_lists = (tr_id, vl_id, te_id)
        
    # -----------------------------------------------
    #      ML model configs
    # -----------------------------------------------
    if args['ml']=='lgb':
        # LGBM regressor model definition
        import lightgbm as lgb
        framework = 'lightgbm'
        ml_model_def = lgb.LGBMRegressor
        mltype = 'reg'
        ml_init_kwargs = { 'n_estimators': 100, 'max_depth': -1,
                           'learning_rate': 0.1, 'num_leaves': 31,
                           'n_jobs': 8, 'random_state': None }
        ml_fit_kwargs = {'verbose': False, 'early_stopping_rounds': 10}
        keras_callbacks_def = None
        keras_clr_kwargs = None

    elif args['ml']=='keras':
        # Keras model def (reg_go)
        from models.reg_go_model import reg_go_model_def, reg_go_callback_def
        framework = 'keras'
        ml_model_def = reg_go_model_def
        keras_callbacks_def = reg_go_callback_def
        mltype = 'reg'
        ml_init_kwargs = {'input_dim': xdata.shape[1], 'dr_rate': 0.1}

        ml_fit_kwargs = {'epochs': args['epoch'], 'batch_size': 32, 'verbose': 1}
        keras_clr_kwargs = {}
        # keras_clr_kwargs = {'mode': 'trng1', 'base_lr': 0.00005, 'max_lr': 0.0005, 'gamma': None}
        # keras_clr_kwargs = {'mode': 'exp', 'base_lr': 0.00005, 'max_lr': 0.0005, 'gamma': 0.999994}

    # -----------------------------------------------
    #      Learning curve 
    # -----------------------------------------------        
    # LC args
    lc_init_args = { 'cv_lists': cv_lists,
                     'n_splits': args['n_splits'], 'mltype': mltype,
                     'lc_step_scale': args['lc_step_scale'], 'lc_sizes': args['lc_sizes'],
                     'min_size': args['min_size'], 'max_size': args['max_size'],
                     'outdir': rout, 'lc_sizes_arr': args['lc_sizes_arr'],
                     'print_fn': print_fn}
                    
    lc_trn_args = { 'framework': framework,
                    'n_jobs': args['n_jobs'], 
                    'ml_model_def': ml_model_def,
                    'ml_init_args': ml_init_kwargs,
                    'ml_fit_args': ml_fit_kwargs,
                    'keras_callbacks_def': keras_callbacks_def,
                    'keras_clr_args': keras_clr_kwargs }

    # LC object
    lc_obj = LearningCurve( X=xdata, Y=ydata, meta=meta, **lc_init_args )

    if args['hp_file'] is None:
        # The regular workflow where all subsets are trained with the same HPs
        lc_scores = lc_obj.trn_learning_curve( **lc_trn_args )
    else:
        # The workflow follows PS-HPO where we a the set HPs per subset.
        # In this case we need to call the trn_learning_curve() method for
        # every subset with appropriate HPs. We'll need to update lc_sizes_arr
        # for every run of trn_learning_curve().
        fpath = verify_dirpath(args['hp_file'])
        hp = pd.read_csv(fpath)
        hp.to_csv(rout/'hpo_ps.csv', index=False)

        # Params to update based on framework
        if framework == 'lightgbm':
            prm_names = ['gbm_trees', 'gbm_max_depth', 'gbm_lr', 'gbm_leaves']
        elif framework == 'sklearn':
            prm_names = ['rf_trees']
        elif framework == 'keras':
            prm_names = ['dr_rate', 'opt', 'lr', 'batchnorm', 'batch_size']

        # Params of interest
        df_print = hp[ prm_names + ['tr_size', 'mean_absolute_error'] ]
        print_fn( df_print )

        # Find the intersect btw available and requested tr sizes
        tr_sizes = list( set(lc_obj.tr_sizes).intersection(set(hp['tr_size'].unique())) )
        print_fn('\nIntersect btw available and requested tr sizes: {}'.format( tr_sizes ))

        lc_scores = []
        for sz in tr_sizes:
            prm = hp[hp['tr_size']==sz]
            lrn_crv_init_args['lc_sizes_arr'] = [sz]
            lc_obj.tr_sizes = [sz] 
            
            # Update model_init and model_fit params
            prm = prm.to_dict(orient='records')[0]  # unroll single-raw df into dict
                
            # Update args
            print_fn('\nUpdate args for tr size {}'.format(sz))
            print_fn( df_print[ df_print['tr_size']==sz ] )
            for n in prm_names:
                print_fn('{}: set to {}'.format(n, prm[n]))
                args[n] = prm[n]

            ml_init_kwargs, ml_fit_kwargs = get_model_args(args)
            lc_trn_args['ml_init_args'] = ml_init_kwargs
            lc_trn_args['ml_fit_args'] = ml_fit_kwargs

            per_subset_scores = lc_obj.trn_learning_curve( **lc_trn_args )
            lc_scores.append( per_subset_scores )

        # Concat per-subset scores 
        lc_scores = pd.concat(lc_scores, axis=0)

        # Save tr, vl, te separently
        lc_scores[ lc_scores['set']=='tr' ].to_csv( rout/'tr_lc_scores.csv', index=False) 
        lc_scores[ lc_scores['set']=='vl' ].to_csv( rout/'vl_lc_scores.csv', index=False) 
        lc_scores[ lc_scores['set']=='te' ].to_csv( rout/'te_lc_scores.csv', index=False) 

    # Dump all scores
    lc_scores.to_csv( rout/'lc_scores.csv', index=False)

    # Load results and plot
    kwargs = {'tr_set': 'te', 'xtick_scale': 'log2', 'ytick_scale': 'log2'}
    lc_plots.plot_lc_many_metric( lc_scores, outdir=rout, **kwargs )
    # kwargs = {'tr_set': 'te', 'xtick_scale': 'linear', 'ytick_scale': 'linear'}
    # lc_plots.plot_lc_many_metric( lc_scores, outdir=rout, **kwargs )
    
    # Dump args
    dump_dict(args, outpath=rout/'args.txt')
    
    # ====================================
    """
    if args['plot_fit']:
        figsize = (7, 5.5)
        metric_name = 'mean_absolute_error'
        xtick_scale, ytick_scale = 'log2', 'log2'
        plot_args = {'metric_name': metric_name, 'xtick_scale': xtick_scale, 'ytick_scale': xtick_scale, 'figsize': figsize}

        scores_te = lrn_crv_scores[(lrn_crv_scores.metric==metric_name) & (lrn_crv_scores.set=='te')].sort_values('tr_size').reset_index(drop=True)

        # -----  Finally plot power-fit  -----
        tot_pnts = len(scores_te['tr_size'])
        n_pnts_fit = 8 # Number of points to use for curve fitting starting from the largest size

        y_col_name = 'fold0'
        ax = None

        ax = lc_plots.plot_lrn_crv_new(
                x=scores_te['tr_size'][0:], y=scores_te[y_col_name][0:],
                ax=ax, ls='', marker='v', alpha=0.7,
                **plot_args, label='Raw Data')

        size_min_idx = 0 if tot_pnts < n_pnts_fit else tot_pnts - n_pnts_fit

        ax, _, gof = lc_plots.plot_lrn_crv_power_law(
                x=scores_te['tr_size'][size_min_idx:], y=scores_te[y_col_name][size_min_idx:],
                **plot_args, plot_raw=False, ax=ax, alpha=1 );

        ax.legend(frameon=True, fontsize=10, loc='best')
        plt.tight_layout()
        plt.savefig(args['outdir']/f'power_law_fit_{metric_name}.png')

        # -----  Extrapolation  -----
        n_pnts_ext = 1 # Number of points to extrapolate to
        n_pnts_fit = 6 # Number of points to use for curve fitting starting from the largest size

        tot_pnts = len(scores_te['tr_size'])
        m0 = tot_pnts - n_pnts_ext
        size_min_idx = m0 - n_pnts_fit

        ax = None

        # Plot of all the points
        ax = lc_plots.plot_lrn_crv_new(
                x=scores_te['tr_size'][0:size_min_idx], y=scores_te[y_col_name][0:size_min_idx],
                ax=ax, ls='', marker='v', alpha=0.8, color='k',
                **plot_args, label='Excluded Points')

        # Plot of all the points
        ax = lc_plots.plot_lrn_crv_new(
                x=scores_te['tr_size'][size_min_idx:m0], y=scores_te[y_col_name][size_min_idx:m0],
                ax=ax, ls='', marker='*', alpha=0.8, color='g',
                **plot_args, label='Included Points')

        # Extrapolation
        ax, _, mae_et = lc_plots.lrn_crv_power_law_extrapolate(
                x=scores_te['tr_size'][size_min_idx:], y=scores_te[y_col_name][size_min_idx:],
                n_pnts_ext=n_pnts_ext,
                **plot_args, plot_raw_it=False, label_et='Extrapolation', ax=ax );

        ax.legend(frameon=True, fontsize=10, loc='best')
        plt.savefig(args['outdir']/f'power_law_ext_{metric_name}.png')
    """
    # ====================================
    
    if (time()-t0)//3600 > 0:
        print_fn('Runtime: {:.1f} hrs'.format( (time()-t0)/3600) )
    else:
        print_fn('Runtime: {:.1f} min'.format( (time()-t0)/60) )
    
    print_fn('Done.')
    lg.kill_logger()
    del xdata, ydata

    # This is required for HPO via UPF workflow on Theta HPC
    return lc_scores[(lc_scores['metric'] == args['hpo_metric']) & (lc_scores['set'] == 'te')].values[0][3]


def main(args):
    args = parse_args(args)
    args = vars(args)
    score = run(args)
    return score
    

if __name__ == '__main__':
    main(sys.argv[1:])


