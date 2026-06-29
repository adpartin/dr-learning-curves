"""
This "new" version of the code uses a different dataframe for descriptors:
'pan_drugs_dragon7_descriptors.tsv' instead of 'combined_pubchem_dragon7_descriptors.tsv'
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from time import time
from pprint import pformat

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer, KNNImputer

# github.com/mtg/sms-tools/issues/36
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Utils
from utils.classlogger import Logger
from utils.utils import dump_dict, get_print_func, dropna
from utils.impute import impute_values
from utils.resample import flatten_dist
from ml.scale import scale_fea
from ml.data import extract_subset_fea

# File path
filepath = Path(__file__).resolve().parent

# Settings
na_values = ['na', '-', '']
fea_prfx_dct = {'ge': 'ge_', 'cnv': 'cnv_', 'snp': 'snp_',
                'dd': 'dd_', 'fng': 'fng_'}


def create_basename(args):
    """ Name to characterize the data. Can be used for dir name and file name. """
    ls = args.drug_fea + args.cell_fea
    if args.src is None:
        name = '.'.join(ls)
    else:
        src_names = '_'.join(args.src)
        name = '.'.join([src_names] + ls)
    name = 'data.' + name
    return name


def create_outdir(outdir, args):
    """ Creates output dir. """
    basename = create_basename(args)
    outdir = Path(outdir, basename)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def groupby_src_and_print(df, print_fn=print):
    print_fn(df.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())


def add_fea_prfx(df, prfx: str, id0: int):
    """ Add prefix feature columns. """
    return df.rename(columns={s: prfx+str(s) for s in df.columns[id0:]})


def load_rsp(fpath, src=None, r2fit_th=None, print_fn=print):
    """ Load drug response data. """
    rsp = pd.read_csv(fpath, sep='\t', na_values=na_values)
    rsp.drop(columns='STUDY', inplace=True)  # gives error when saves in 'parquet' format
    # print(rsp.dtypes)

    print_fn('\nAll samples (original).')
    print_fn(f'rsp.shape {rsp.shape}')
    groupby_src_and_print(rsp, print_fn=print_fn)
    print_fn(rsp.SOURCE.value_counts())

    # Drop bad samples
    if r2fit_th is not None:
        # Yitan
        # TODO: check this (may require a more rigorous filtering)
        # print_fn('\n\nDrop bad samples ...')
        # id_drop = (rsp['AUC'] == 0) & (rsp['EC50se'] == 0) & (rsp['R2fit'] == 0)
        # rsp = rsp.loc[~id_drop,:]
        # print_fn(f'Dropped {sum(id_drop)} rsp data points.')
        # print_fn(f'rsp.shape {rsp.shape}')
        print_fn('\nDrop samples with low R2fit.')
        print_fn('Samples with bad fit.')
        id_drop = rsp['R2fit'] <= r2fit_th
        rsp_bad_fit = rsp.loc[id_drop, :].reset_index(drop=True)
        groupby_src_and_print(rsp_bad_fit, print_fn=print_fn)
        print_fn(rsp_bad_fit.SOURCE.value_counts())

        print_fn('\nSamples with good fit.')
        rsp = rsp.loc[~id_drop, :].reset_index(drop=True)
        groupby_src_and_print(rsp, print_fn=print_fn)
        print_fn(rsp.SOURCE.value_counts())
        print_fn(f'Dropped {sum(id_drop)} rsp data points.')

    rsp['SOURCE'] = rsp['SOURCE'].apply(lambda x: x.lower())

    if src is not None:
        print_fn('\nExtract specific sources.')
        rsp = rsp[rsp['SOURCE'].isin(src)].reset_index(drop=True)

    rsp['AUC_bin'] = rsp['AUC'].map(lambda x: 0 if x > 0.5 else 1)
    rsp.replace([np.inf, -np.inf], value=np.nan, inplace=True)

    print_fn(f'rsp.shape {rsp.shape}')
    groupby_src_and_print(rsp, print_fn=print_fn)
    return rsp


def load_ge(fpath, print_fn=print, float_type=np.float32):
    """ Load RNA-Seq data. """
    print_fn(f'\nLoad RNA-Seq ... {fpath}')
    ge = pd.read_csv(fpath, sep='\t', na_values=na_values)
    ge.rename(columns={'Sample': 'CELL'}, inplace=True)

    fea_id0 = 1
    ge = add_fea_prfx(ge, prfx=fea_prfx_dct['ge'], id0=fea_id0)

    if sum(ge.isna().sum() > 0):
        # ge = impute_values(ge, print_fn=print_fn)

        print_fn('Columns with NaNs: {}'.format( sum(ge.iloc[:, fea_id0:].isna().sum() > 0) ))
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        # imputer = KNNImputer(missing_values=np.nan, n_neighbors=5,
        #                      weights='uniform', metric='nan_euclidean',
        #                      add_indicator=False)
        ge.iloc[:, fea_id0:] = imputer.fit_transform(ge.iloc[:, fea_id0:].values)
        print_fn('Columns with NaNs: {}'.format( sum(ge.iloc[:, fea_id0:].isna().sum() > 0) ))

    # Cast features (casting to float16 changes the shape. why?)
    ge = ge.astype(dtype={c: float_type for c in ge.columns[fea_id0:]})
    print_fn(f'ge.shape {ge.shape}')
    return ge


def load_dd(fpath, print_fn=print, dropna_th=0.1, float_type=np.float32, src=None):
    """ Load drug descriptors. """
    print_fn(f'\nLoad descriptors ... {fpath}')
    dd = pd.read_csv(fpath, sep='\t', na_values=na_values)
    dd.rename(columns={'ID': 'DRUG'}, inplace=True)

    # dd = add_fea_prfx(dd, prfx=fea_prfx_dct['dd'], id0=fea_id0)

    if 'nci60' in src:
        dd = dropna(dd, axis=0, th=dropna_th)
        fea_id0 = 2
    else:
        fea_id0 = 4

    if sum(dd.isna().sum() > 0):
        print_fn('Columns with all NaN values: {}'.format(
            sum(dd.isna().sum(axis=0).sort_values(ascending=False) == dd.shape[0])))
        print_fn('Columns with NaNs: {}'.format( sum(dd.iloc[:, fea_id0:].isna().sum() > 0) ))
        imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
        # imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        # imputer = KNNImputer(missing_values=np.nan, n_neighbors=5,
        #                      weights='uniform', metric='nan_euclidean',
        #                      add_indicator=False)
        dd.iloc[:, fea_id0:] = imputer.fit_transform(dd.iloc[:, fea_id0:].values)
        print_fn('Columns with NaNs: {}'.format( sum(dd.iloc[:, fea_id0:].isna().sum() > 0) ))

    # Cast features
    dd = dd.astype(dtype={c: float_type for c in dd.columns[fea_id0:]})
    print_fn(f'dd.shape {dd.shape}')
    return dd


def plot_dd_na_dist(dd, savepath=None):
    """ Plot distbirution of na values in drug descriptors. """
    fig, ax = plt.subplots()
    sns.distplot(dd.isna().sum(axis=0)/dd.shape[0], bins=100, kde=False, hist_kws={'alpha': 0.7})
    plt.xlabel('Ratio of total NA values in a descriptor to the total drug count')
    plt.ylabel('Total # of descriptors with the specified NA ratio')
    plt.title('Histogram of descriptors based on ratio of NA values')
    plt.grid(True)
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight') # dpi=200
    else:
        plt.savefig('dd_hist_ratio_of_na.png', bbox_inches='tight') # dpi=200


def plot_rsp_dists(rsp, rsp_cols, savepath=None):
    """ Plot distributions of all response variables.
    Args:
        rsp : df of response values
        rsp_cols : list of col names
        savepath : full path to save the image
    """
    ncols = 4
    nrows = int(np.ceil(len(rsp_cols)/ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False, figsize=(10, 10))
    for i, ax in enumerate(axes.ravel()):
        if i >= len(rsp_cols):
            fig.delaxes(ax)  # delete un-used ax
        else:
            target_name = rsp_cols[i]
            x = rsp[target_name].copy()
            x = x[~x.isna()].values
            sns.distplot(x, bins=100, kde=True, ax=ax, label=target_name,  # fit=norm,
                         kde_kws={'color': 'k', 'lw': 0.4, 'alpha': 0.8},
                         hist_kws={'color': 'b', 'lw': 0.4, 'alpha': 0.5})
            ax.tick_params(axis='both', which='major', labelsize=7)
            txt = ax.yaxis.get_offset_text(); txt.set_size(7)  # adjust exponent fontsize in xticks
            txt = ax.xaxis.get_offset_text(); txt.set_size(7)
            ax.legend(fontsize=5, loc='best')
            ax.grid(True)

    # plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')  # dpi=200
    else:
        plt.savefig('rsp_dists.png', bbox_inches='tight')


def parse_args(args):
    parser = argparse.ArgumentParser(description='Create ML dataframe.')

    parser.add_argument('--rsp_path',
                        type=str,
                        help='Path to drug response file.')
    parser.add_argument('--drug_path',
                        type=str,
                        help='Path to drug features file.')
    parser.add_argument('--cell_path',
                        type=str,
                        help='Path to cell features file.')
    parser.add_argument('--r2fit_th',
                        type=float,
                        default=0.5,
                        help='Drop drug response values with R-square fit \
                        less than this value (Default: 0.5).')

    parser.add_argument('--drug_fea',
                        type=str,
                        nargs='+',
                        choices=['dd'],
                        default=['dd'],
                        help='Default: [dd].')
    parser.add_argument('--cell_fea',
                        type=str,
                        nargs='+',
                        choices=['ge'],
                        default=['ge'],
                        help='Default: [ge].')
    parser.add_argument('--gout',
                        type=str,
                        help='Default: ...')
    parser.add_argument('--dropna_th',
                        type=float,
                        default=0,
                        help='Default: 0')
    parser.add_argument('--src',
                        nargs='+',
                        default=None,
                        choices=['ccle', 'gcsi', 'gdsc', 'gdsc1', 'gdsc2', 'ctrp', 'nci60'],
                        help='Data sources to extract (default: None).')

    parser.add_argument('--n_samples',
                        type=int,
                        default=None,
                        help='Number of docking scores to get into the ML df (default: None).')
    parser.add_argument('--flatten',
                        action='store_true',
                        help='Flatten the distribution of response values (default: False).')
    parser.add_argument('-t', '--trg_name',
                        default='AUC',
                        type=str,
                        choices=['AUC'],
                        help='Name of target variable (default: AUC).')

    args = parser.parse_args(args)
    return args


def run(args):
    # import ipdb; ipdb.set_trace(context=5)
    t0 = time()
    rsp_cols = ['AUC', 'AUC1', 'EC50', 'EC50se', 'R2fit',
                'Einf', 'IC50', 'HS', 'AAC1', 'DSS1']
    outdir = create_outdir(args.gout, args)

    # -----------------------------------------------
    #     Logger
    # -----------------------------------------------
    lg = Logger(outdir/'gen.df.log')
    print_fn = get_print_func(lg.logger)
    print_fn(f'File path: {filepath}')
    print_fn(f'\n{pformat(vars(args))}')
    dump_dict(vars(args), outpath=outdir/'gen.df.args')

    # -----------------------------------------------
    #     Load response data and features
    # -----------------------------------------------
    rsp = load_rsp(args.rsp_path, src=args.src, r2fit_th=args.r2fit_th,
                   print_fn=print_fn)
    ge = load_ge(args.cell_path, print_fn=print_fn, float_type=np.float32)
    dd = load_dd(args.drug_path, dropna_th=args.dropna_th, print_fn=print_fn,
                 float_type=np.float32, src=args.src)

    # -----------------------------------------------
    #     Merge data
    # -----------------------------------------------
    print_fn('\n{}'.format('-' * 40))
    print_fn('Start merging response with other dfs.')
    print_fn('-' * 40)
    data = rsp

    # Merge with ge
    print_fn('\nMerge with expression (ge).')
    data = pd.merge(data, ge, on='CELL', how='inner')
    groupby_src_and_print(data, print_fn=print_fn)
    del ge

    # Merge with dd
    print_fn('\nMerge with descriptors (dd).')
    data = pd.merge(data, dd, on='DRUG', how='inner')
    groupby_src_and_print(data, print_fn=print_fn)
    del dd

    # Sample
    if (args.n_samples is not None):
        print_fn('\nSample the final dataset.')
        if args.flatten:
            data = flatten_dist(df=data, n=args.n_samples, score_name=args.trg_name)
        else:
            if args.n_samaples <= data.shape[0]:
                data = data.sample(n=args.n_samples, replace=False, random_state=0)
        print_fn(f'data.shape {data.shape}\n')

    # Memory usage
    print_fn('\nTidy dataframe: {:.1f} GB'.format(sys.getsizeof(data)/1e9))
    for fea_name, fea_prfx in fea_prfx_dct.items():
        cols = [c for c in data.columns if fea_prfx in c]
        aa = data[cols]
        mem = 0 if aa.shape[1] == 0 else sys.getsizeof(aa)/1e9
        print_fn('Memory occupied by {} features: {} ({:.1f} GB)'.format(
            fea_name, len(cols), mem))

    # Plot histograms of target variables
    plot_rsp_dists(data, rsp_cols=rsp_cols, savepath=outdir/'rsp_dists.png')

    # -----------------------------------------------
    #   Save data
    # -----------------------------------------------
    # Save data
    print_fn('\nSave dataframe.')
    fname = create_basename(args)
    fpath = outdir/(fname + '.parquet')
    data.to_parquet(fpath)

    # Load data
    print_fn('Load dataframe.')
    data_fromfile = pd.read_parquet(fpath)

    # Check that the saved data is the same as original one
    print_fn(f'Loaded df is same as original: {data.equals(data_fromfile)}')

    print_fn('\n{}'.format('-' * 70))
    print_fn(f'Dataframe filepath:\n{fpath.resolve()}')
    print_fn('-' * 70)

    # -------------------------------------------------------
    print_fn('\nRuntime: {:.1f} mins'.format((time()-t0)/60))
    print_fn('Done.')
    lg.kill_logger()


def main(args):
    args = parse_args(args)
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
