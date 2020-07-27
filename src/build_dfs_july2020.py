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
from utils.utils import load_data, dump_dict, get_print_func
from utils.impute import impute_values
from ml.scale import scale_fea
from ml.data import extract_subset_fea

# File path
filepath = Path(__file__).resolve().parent

# Default settings
# DATADIR = Path( filepath/'../data/raw' ).resolve()
# OUTDIR = Path( DATADIR/'../ml.dfs' ).resolve()
# os.makedirs(OUTDIR, exist_ok=True)

# -----------
# PreJuly2020
# -----------
# RSP_FILENAME = 'combined_single_response_agg'  # reposne data
# # RSP_FILENAME_CHEM = 'chempartner_single_response_agg'  # reposne data
# DSC_FILENAME = 'pan_drugs_dragon7_descriptors.tsv'  # drug descriptors data (new)
# DSC_NCI60_FILENAME = 'NCI60_dragon7_descriptors.tsv'  # drug descriptors data (new)
# DRUG_META_FILENAME = 'drug_info'
# CELL_META_FILENAME = 'combined_cancer_types'
# # CELL_META_FILENAME = 'combined_metadata_2018May.txt'

# -----------
# July2020
# -----------
# RSP_FILENAME = 'combined_single_response_agg'  # reposne data
# # RSP_FILENAME_CHEM = 'chempartner_single_response_agg'  # reposne data
# DSC_FILENAME = 'pan_drugs_dragon7_descriptors.tsv'  # drug descriptors data (new)
# DSC_NCI60_FILENAME = 'NCI60_dragon7_descriptors.tsv'  # drug descriptors data (new)
# DRUG_META_FILENAME = 'drug_info'
# CELL_META_FILENAME = 'combined_cancer_types'
# # CELL_META_FILENAME = 'combined_metadata_2018May.txt'



# Settings
na_values = ['na', '-', '']
fea_prfx_dct = {'ge': 'ge_', 'cnv': 'cnv_', 'snp': 'snp_',
                'dd': 'dd_', 'fng': 'fng_'}

# prfx_dtypes = {'ge': np.float32, 'cnv': np.int8, 'snp': np.int8,
#                'dd': np.float32, 'fng': np.int8}


def create_basename(args):
    """ Name to characterize the data. Can be used for dir name and file name. """
    ls = args['drug_fea'] + args['cell_fea']  # + [args['ge_norm']]
    if args['src'] is None:
        name = '.'.join( ls )
    else:
        src_names = '_'.join( args['src'] )
        name = '.'.join( [src_names] + ls )
    name = 'data.' + name
    return name


def create_outdir(outdir, args):
    """ Creates output dir. """
    basename = create_basename(args)
    outdir = Path(outdir, basename)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def groupby_src_and_print(df, print_fn=print):        
    print_fn( df.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index() )


def load_rsp(fpath, src=None, r2fit_th=None, print_fn=print):
    """ Load drug response data. """
    # print_fn(f'\nLoad response from ... {DATADIR/RSP_FILENAME}')
    # rsp = pd.read_table(DATADIR/RSP_FILENAME, sep='\t', na_values=na_values, warn_bad_lines=True)
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
        rsp_bad_fit = rsp.loc[id_drop,:].reset_index(drop=True)
        groupby_src_and_print(rsp_bad_fit, print_fn=print_fn)
        print_fn(rsp_bad_fit.SOURCE.value_counts())

        print_fn('\nSamples with good fit.')
        rsp = rsp.loc[~id_drop,:].reset_index(drop=True)
        groupby_src_and_print(rsp, print_fn=print_fn)
        print_fn(rsp.SOURCE.value_counts())
        print_fn(f'Dropped {sum(id_drop)} rsp data points.')

    rsp['SOURCE'] = rsp['SOURCE'].apply(lambda x: x.lower())

    if src is not None:
        print_fn('\nExtract specific sources.')
        rsp = rsp[rsp['SOURCE'].isin(src)].reset_index(drop=True)

    rsp.replace([np.inf, -np.inf], value=np.nan, inplace=True) # Replace -inf and inf with nan

    print_fn(f'rsp.shape {rsp.shape}')
    groupby_src_and_print(rsp, print_fn=print_fn)
    return rsp


def load_ge(fpath, print_fn=print, float_type=np.float32):
    """ Load RNA-Seq data. """
    print_fn(f'\nLoad RNA-Seq ... {fpath}')
    ge = pd.read_csv(fpath, sep='\t', na_values=na_values)
    ge.rename(columns={'Sample': 'CELL'}, inplace=True)

    fea_start_id = 1
    col_rename = {c: fea_prfx_dct['ge']+c for c in ge.columns[fea_start_id:]
                  if fea_prfx_dct['ge'] not in c}
    ge = ge.rename(columns=col_rename)  # prefix ge gene names

    if sum(ge.isna().sum() > 0):
        # ge = impute_values(ge, print_fn=print_fn)

        print('Columns with NaNs: {}'.format( sum(ge.iloc[:, fea_start_id:].isna().sum() > 0) ))
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        # imputer = KNNImputer(missing_values=np.nan, n_neighbors=5,
        #                      weights='uniform', metric='nan_euclidean',
        #                      add_indicator=False)
        ge.iloc[:, fea_start_id:] = imputer.fit_transform(ge.iloc[:, fea_start_id:].values)
        print('Columns with NaNs: {}'.format( sum(ge.iloc[:, fea_start_id:].isna().sum() > 0) ))

    # Cast features (casting to float16 changes the shape. why?)
    ge = ge.astype(dtype={c: float_type for c in ge.columns[fea_start_id:]})
    print_fn(f'ge.shape {ge.shape}')
    return ge


def load_dd(fpath, print_fn=print, dropna_th=0.4, float_type=np.float32,
            src=None, impute=True, plot=True, outdir=None):
    """ Load drug descriptors. """
    # if 'nci60' in src:
    #     fpath = DATADIR/DSC_NCI60_FILENAME
    # else:
    #     fpath = DATADIR/DSC_FILENAME
    print_fn(f'\nLoad drug descriptors ... {fpath}')
    dd = pd.read_csv(fpath, sep='\t', na_values=na_values)
    dd.rename(columns={'ID': 'DRUG'}, inplace=True)

    # Prefix drug desc names
    fea_start_id = 4
    col_rename = {c: fea_prfx_dct['dd']+c for c in dd.columns[fea_start_id:]
                  if fea_prfx_dct['dd'] not in c}
    dd = dd.rename(columns=col_rename)


    # if 'nci60' in src:
    #     # For NCI60, extract specific columns that were chosen for CTRP
    #     col_names = pd.read_csv(OUTDIR/'dd_col_names.csv').columns.tolist()
    #     dd = dd[col_names]

    #     # Filter sample with too many NaN values
    #     dd = dropna(dd, axis=0, max_na=0)

    # else:
    #     # ------------------
    #     # Filter descriptors
    #     # ------------------
    #     # # dd.nunique(dropna=True).value_counts()
    #     # # dd.nunique(dropna=True).sort_values()
    #     # print_fn('Drop descriptors with too many NA values ...')
    #     # if plot and (outdir is not None):
    #     #     plot_dd_na_dist(dd=dd, savepath=Path(outdir, 'dd_hist_ratio_of_na.png'))
    #     # dd = dropna(dd, axis=1, th=dropna_th)
    #     # print_fn(f'dd.shape {dd.shape}')
    #     # # dd.isna().sum().sort_values(ascending=False)

    #     # There are descriptors where there is a single unique value excluding NA (drop those)
    #     print_fn('Drop descriptors that have a single unique value (excluding NAs) ...')
    #     # col_idx = dd.nunique(dropna=True).values==1  # takes too long for large dataset
    #     # dd = dd.iloc[:, ~col_idx]
    #     # TODO this filtering replaces the filtering above!
    #     dd_names = dd.iloc[:, 0]
    #     dd_fea = dd.iloc[:, fea_start_id:].copy()
    #     col_idx = dd_fea.std(axis=0, skipna=True, numeric_only=True).values == 0
    #     dd_fea = dd_fea.iloc[:, ~col_idx]
    #     dd = pd.concat([dd_names, dd_fea], axis=1)
    #     print_fn(f'dd.shape {dd.shape}')

    #     # There are still lots of descriptors which have only a few unique values
    #     # we can categorize those values. e.g.: 564 descriptors have only 2 unique vals,
    #     # and 154 descriptors have only 3 unique vals, etc.
    #     # todo: use utility code from p1h_alex/utils/data_preproc.py that transform those
    #     # features into categorical and also applies an appropriate imputation.
    #     # dd.nunique(dropna=true).value_counts()[:10]
    #     # dd.nunique(dropna=true).value_counts().sort_index()[:10]

    #     # Save feature names
    #     dd.iloc[:1, :].to_csv(Path(outdir, 'dd_col_names.csv'), index=False)

    # ---------------------
    # Impute missing values
    # ---------------------
    # if impute:
    #     print_fn('Impute NA values ...')
    #     dd = impute_values(data=dd, print_fn=print_fn)

    if sum(dd.isna().sum() > 0):
        print_fn('Columns with all NaN values: {}'.format(
            sum(dd.isna().sum(axis=0).sort_values(ascending=False)==dd.shape[0])))
        print_fn('Columns with NaNs: {}'.format( sum(dd.iloc[:, fea_start_id:].isna().sum() > 0) ))
        imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
        # imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        # imputer = KNNImputer(missing_values=np.nan, n_neighbors=5,
        #                      weights='uniform', metric='nan_euclidean',
        #                      add_indicator=False)
        dd.iloc[:, fea_start_id:] = imputer.fit_transform(dd.iloc[:, fea_start_id:].values)
        print_fn('Columns with NaNs: {}'.format( sum(dd.iloc[:, fea_start_id:].isna().sum() > 0) ))

    # Cast features
    dd = dd.astype(dtype={c: float_type for c in dd.columns[fea_start_id:]})
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


def dropna(df, axis=0, th=0.05, max_na=None):
    """ Drop rows or cols based on the ratio of NA values along the axis.
    Instead of ratio, you can also specify the max number of NA.
    Args:
        th (float) : if the ratio of NA values along the axis is larger that th, then drop all the values
        max_na (int) : specify max allowable number of na (instead of specifying the ratio)
        axis (int) : 0 to drop rows; 1 to drop cols
    """
    # df = df.copy()
    axis = 0 if (axis == 1) else 1

    if max_na is not None:
        assert max_na >= 0, 'max_na must be >=0.'
        idx = df.isna().sum(axis=axis) <= max_na
    else:
        idx = df.isna().sum(axis=axis)/df.shape[axis] <= th

    if axis == 0:
        df = df.iloc[:, idx.values]
    else:
        df = df.iloc[idx.values, :]
    return df


def plot_rsp_dists(rsp, rsp_cols, savepath=None):
    """ Plot distributions of all response variables.
    Args:
        rsp : df of response values
        rsp_cols : list of col names
        savepath : full path to save the image
    """
    ncols = 4
    nrows = int(np.ceil(len(rsp_cols)/ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False, figsize=(10,10))
    for i, ax in enumerate(axes.ravel()):
        if i >= len(rsp_cols):
            fig.delaxes(ax) # delete un-used ax
        else:
            target_name = rsp_cols[i]
            x = rsp[target_name].copy()
            x = x[~x.isna()].values
            sns.distplot(x, bins=100, kde=True, ax=ax, label=target_name, # fit=norm, 
                        kde_kws={'color': 'k', 'lw': 0.4, 'alpha': 0.8},
                        hist_kws={'color': 'b', 'lw': 0.4, 'alpha': 0.5})
            ax.tick_params(axis='both', which='major', labelsize=7)
            txt = ax.yaxis.get_offset_text(); txt.set_size(7) # adjust exponent fontsize in xticks
            txt = ax.xaxis.get_offset_text(); txt.set_size(7)
            ax.legend(fontsize=5, loc='best')
            ax.grid(True)

    # plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight') # dpi=200
    else:
        plt.savefig('rsp_dists.png', bbox_inches='tight')


def parse_args(args):
    parser = argparse.ArgumentParser(description='Create tidy data.')

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
                        default=0.7,
                        help='Drop drug response values with R-square fit \
                        less than this value (Default: 0.7).')

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

    args = parser.parse_args(args)
    return args


def run(args):
    # import pdb; pdb.set_trace()
    t0 = time()
    rsp_cols = ['AUC', 'AUC1', 'EC50', 'EC50se', 'R2fit',
                'Einf', 'IC50', 'HS', 'AAC1', 'DSS1']

    # -----------------------------------------------
    #     Create outdir
    # -----------------------------------------------
    outdir = create_outdir(args['gout'], args)
    args['outdir'] = str(outdir)

    # -----------------------------------------------
    #     Logger
    # -----------------------------------------------
    lg = Logger(outdir/'gen.df.log')
    print_fn = get_print_func(lg.logger)
    print_fn(f'File path: {filepath}')
    print_fn(f'\n{pformat(args)}')
    dump_dict(args, outpath=outdir/'gen.df.args')

    # -----------------------------------------------
    #     Load response data, and features
    # -----------------------------------------------
    rsp = load_rsp(args['rsp_path'], src=args['src'], r2fit_th=args['r2fit_th'],
                   print_fn=print_fn)
    ge = load_ge(args['cell_path'], print_fn=print_fn, float_type=np.float32)
    dd = load_dd(args['drug_path'], dropna_th=args['dropna_th'], print_fn=print_fn,
                 float_type=np.float32, src=args['src'], outdir=outdir)

    # -----------------------------------------------
    #     Merge data
    # -----------------------------------------------
    print_fn('\n{}'.format('-'*40))
    print_fn('Start merging response with other dataframes ...')
    print_fn('-'*40)
    data = rsp

    # Merge with ge
    print_fn('\nMerge with expression (ge) ...')
    data = pd.merge(data, ge, on='CELL', how='inner')  # inner join to keep samples that have ge
    print_fn(f'data.shape {data.shape}\n')
    groupby_src_and_print(data, print_fn=print_fn)
    del ge

    # Merge with dd
    print_fn('\nMerge with descriptors (dd) ...')
    data = pd.merge(data, dd, on='DRUG', how='inner')  # inner join to keep samples that have dd
    print_fn(f'data.shape {data.shape}\n')
    groupby_src_and_print(data, print_fn=print_fn)
    del dd

    # Sample from NCI60 (specify the max size)
    # max_sz = 500000
    # max_sz = 650000
    # max_sz = 700000
    max_sz = 750000
    # max_sz = 900000
    if ('nci60' in args['src']) and (data.shape[0] > max_sz):
        print_fn('\nSample the final dataset ...')
        data = data.sample(n=max_sz, random_state=0)
        print_fn(f'data.shape {data.shape}\n')

    # Memory usage
    print_fn('\nTidy dataframe: {:.2f} GB'.format(sys.getsizeof(data)/1e9))
    for fea_name, fea_prfx in fea_prfx_dct.items():
        cols = [c for c in data.columns if fea_prfx in c]
        tmp = data[cols]
        mem = 0 if tmp.shape[1] == 0 else sys.getsizeof(tmp)/1e9
        print_fn('Memory occupied by {} features: {} ({:.2f} GB)'.format(
            fea_name, len(cols), mem))

    print_fn('\nRuntime: {:.1f} mins'.format( (time()-t0)/60) )

    # -----------------------------------------------
    #   Plot rsp distributions
    # -----------------------------------------------
    # Plot distributions of target variables
    plot_rsp_dists(data, rsp_cols=rsp_cols, savepath=outdir/'rsp_dists.png')

    # Plot distribution of a single target
    # target_name = 'EC50se'
    # fig, ax = plt.subplots()
    # x = rsp[target_name].copy()
    # x = x[~x.isna()].values
    # sns.distplot(x, bins=100, ax=ax)
    # plt.savefig(os.path.join(OUTDIR, target_name+'.png'), bbox_inches='tight')

    # -----------------------------------------------
    #   Finally save data
    # -----------------------------------------------
    # Save data
    print_fn('\nSave tidy dataframe ...')
    fname = create_basename(args)
    fpath = outdir/(fname+'.parquet')
    data.to_parquet(fpath)

    # Load data
    print_fn('Load tidy dataframe ...')
    data_fromfile = pd.read_parquet(fpath)

    # Check that the saved data is the same as original one
    print_fn(f'Loaded dataframe is same as original: {data.equals(data_fromfile)}')

    # --------------------------------------------------------
    print_fn('\n{}'.format('-' * 80))
    print_fn(f'Tidy data filepath:\n{os.path.abspath(fpath)}')
    print_fn('-' * 80)
    lg.kill_logger()


def main(args):
    args = parse_args(args)
    args = vars(args)
    run(args)
    print('Done.')


if __name__ == '__main__':
    main(sys.argv[1:])


