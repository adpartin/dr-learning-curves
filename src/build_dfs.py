"""
This "new" version of the code uses a different dataframe for descriptors:
'pan_drugs_dragon7_descriptors.tsv' instead of 'combined_pubchem_dragon7_descriptors.tsv'
"""
from __future__ import division, print_function

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from time import time
from pprint import pprint, pformat

import sklearn
import numpy as np
import pandas as pd

# github.com/mtg/sms-tools/issues/36
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# File path
filepath = Path(__file__).resolve().parent

# Utils
from utils.classlogger import Logger
from utils.utils import load_data, dump_dict, get_print_func
from utils.impute import impute_values
from ml.scale import scale_fea
from ml.data import extract_subset_fea


# Default settings
DATADIR = Path( filepath/'../data/raw' ).resolve()
OUTDIR = Path( DATADIR/'../ml.dfs' ).resolve()
os.makedirs(OUTDIR, exist_ok=True)

RSP_FILENAME = 'combined_single_response_agg'  # reposne data
# RSP_FILENAME_CHEM = 'chempartner_single_response_agg'  # reposne data
DSC_FILENAME = 'pan_drugs_dragon7_descriptors.tsv'  # drug descriptors data (new)
DSC_NCI60_FILENAME = 'NCI60_dragon7_descriptors.tsv'  # drug descriptors data (new)
DRUG_META_FILENAME = 'drug_info'
CELL_META_FILENAME = 'combined_cancer_types'
# CELL_META_FILENAME = 'combined_metadata_2018May.txt'


# Settings
na_values = ['na', '-', '']
fea_prfx_dct = {'ge': 'ge_', 'cnv': 'cnv_', 'snp': 'snp_',
                'dd': 'dd_', 'fng': 'fng_'}

prfx_dtypes = {'ge': np.float32, 'cnv': np.int8, 'snp': np.int8,
               'dd': np.float32, 'fng': np.int8}


def create_basename(args):
    """ Name to characterize the data. Can be used for dir name and file name. """
    ls = args['drug_fea'] + args['cell_fea'] + [args['ge_norm']]
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
    outdir = Path(outdir)/basename
    os.makedirs(outdir, exist_ok=True)
    return outdir


def dump_dict(dct, outpath='./dict.txt'):
    """ Dump dict into file. """
    with open( Path(outpath), 'w' ) as file:
        for k in sorted(dct.keys()):
            file.write('{}: {}\n'.format(k, dct[k]))


def groupby_src_and_print(df, print_fn=print):        
    print_fn( df.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index() )


def load_rsp(filename, src=None, keep_bad=False, print_fn=print):
    """ Load drug response data. """
    print_fn(f'\nLoad response from ... {DATADIR/RSP_FILENAME}')
    rsp = pd.read_table(DATADIR/RSP_FILENAME, sep='\t', na_values=na_values, warn_bad_lines=True)
    rsp.drop(columns='STUDY', inplace=True) # gives error when saves in 'parquet' format
    # print(rsp.dtypes)

    print_fn(f'rsp.shape {rsp.shape}')
    groupby_src_and_print(rsp, print_fn=print_fn)        
        
    # Drop bad samples
    if keep_bad is False:
        # Yitan
        # print_fn('\n\nDrop bad samples ...')
        # id_drop = (rsp['AUC'] == 0) & (rsp['EC50se'] == 0) & (rsp['R2fit'] == 0)
        # rsp = rsp.loc[~id_drop,:]
        # print_fn(f'Dropped {sum(id_drop)} rsp data points.')
        # print_fn(f'rsp.shape {rsp.shape}')
        # (ap)
        # TODO: check this (may require a more rigorous filtering)
        print_fn('\nDrop samples with low R2fit ...')
        id_drop = rsp['R2fit'] <= 0
        rsp = rsp.loc[~id_drop,:]
        print_fn(f'Dropped {sum(id_drop)} rsp data points.')

    print_fn('\nExtract specific sources.')
    rsp['SOURCE'] = rsp['SOURCE'].apply(lambda x: x.lower())
    rsp.replace([np.inf, -np.inf], value=np.nan, inplace=True) # Replace -inf and inf with nan

    if src is not None:
        rsp = rsp[rsp['SOURCE'].isin(src)].reset_index(drop=True)

    print_fn(f'rsp.shape {rsp.shape}')
    groupby_src_and_print(rsp, print_fn=print_fn)
    return rsp


def load_ge(datadir, ge_norm, print_fn=print, keep_cells_only=True, float_type=np.float32, impute=True):
    """ Load RNA-Seq data. """
    if ge_norm == 'raw':
        fname = 'combined_rnaseq_data_lincs1000'
    else:
        fname = f'combined_rnaseq_data_lincs1000_{ge_norm}'

    print_fn(f'\nLoad RNA-Seq ... {datadir/fname}')
    ge = pd.read_csv(Path(datadir)/fname, sep='\t', low_memory=False,
                      na_values=na_values, warn_bad_lines=True)
    ge = ge.astype(dtype={c: float_type for c in ge.columns[1:]})  # Cast features
    ge = ge.rename(columns={c: fea_prfx_dct['ge']+c for c in ge.columns[1:] if fea_prfx_dct['ge'] not in c}) # prefix ge gene names
    ge.rename(columns={'Sample': 'CELL'}, inplace=True) # rename cell col name
    # ge = ge.set_index(['CELL'])

    cell_sources = ['ccle', 'gcsi', 'gdsc', 'ctrp', 'nci60']
    idx = ge['CELL'].map(lambda s: s.split('.')[0].lower() in cell_sources)
    ge = ge.loc[idx, :].reset_index(drop=True)

    # Impute missing values
    if impute:
        print_fn('Impute NA values ...')
        ge = impute_values(ge, print_fn=print_fn)

    print_fn(f'ge.shape {ge.shape}')
    return ge


def load_dd(filename, print_fn=print, dropna_th=0.4, float_type=np.float32,
            src=None, impute=True, plot=True, outdir=None):
    """ Load drug descriptors. """
    if 'nci60' in src:
        path = DATADIR/DSC_NCI60_FILENAME
    else:
        path = DATADIR/DSC_FILENAME
    print_fn(f'\nLoad drug descriptors ... {path}')
    dd = pd.read_csv(path, sep='\t', low_memory=False,
                     na_values=na_values, warn_bad_lines=True)

    dd = dd.astype(dtype={c: float_type for c in dd.columns[1:]})  # Cast features
    dd.rename(columns={'NAME': 'DRUG'}, inplace=True)

    # Prefix drug desc names
    dd = dd.rename(columns={c: fea_prfx_dct['dd']+c for c in dd.columns[1:] if fea_prfx_dct['dd'] not in c})

    if 'nci60' in src:
        # For NCI60, extract specific columns that were chosen for CTRP
        col_names = pd.read_csv(OUTDIR/'dd_col_names.csv').columns.tolist()
        dd = dd[col_names]

        # Filter sample with too many NaN values
        dd = dropna(dd, axis=0, max_na=0)

    else:
        # ------------------
        # Filter descriptors
        # ------------------
        # dd.nunique(dropna=True).value_counts()
        # dd.nunique(dropna=True).sort_values()
        print_fn('Drop descriptors with too many NA values ...')
        if plot and (outdir is not None):
            plot_dd_na_dist(dd=dd, savepath=Path(outdir, 'dd_hist_ratio_of_na.png'))
        dd = dropna(dd, axis=1, th=dropna_th)
        print_fn(f'dd.shape {dd.shape}')
        # dd.isna().sum().sort_values(ascending=False)

        # There are descriptors where there is a single unique value excluding NA (drop those)
        print_fn('Drop descriptors that have a single unique value (excluding NAs) ...')
        # col_idx = dd.nunique(dropna=True).values==1 # takes too long for large dataset
        # dd = dd.iloc[:, ~col_idx]
        # TODO this filtering replaces the filtering above!
        dd_names = dd.iloc[:, 0]
        dd_fea = dd.iloc[:, 1:].copy()
        col_idx = dd_fea.std(axis=0, skipna=True, numeric_only=True).values == 0
        dd_fea = dd_fea.iloc[:, ~col_idx]
        dd = pd.concat([dd_names, dd_fea], axis=1)
        print_fn(f'dd.shape {dd.shape}')

        # There are still lots of descriptors which have only a few unique values
        # we can categorize those values. e.g.: 564 descriptors have only 2 unique vals,
        # and 154 descriptors have only 3 unique vals, etc.
        # todo: use utility code from p1h_alex/utils/data_preproc.py that transform those
        # features into categorical and also applies an appropriate imputation.
        # dd.nunique(dropna=true).value_counts()[:10]
        # dd.nunique(dropna=true).value_counts().sort_index()[:10]

        # Save feature names
        dd.iloc[:1, :].to_csv(Path(outdir, 'dd_col_names.csv'), index=False)

    # ---------------------
    # Impute missing values
    # ---------------------
    if impute:
        print_fn('Impute NA values ...')
        dd = impute_values(data=dd, print_fn=print_fn)

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
    parser.add_argument('--ge_norm',
                        type=str,
                        choices=['raw', 'combat', 'source_scale'],
                        default='raw',
                        help='Default: raw.')
    parser.add_argument('--gout',
                        type=str,
                        default=OUTDIR,
                        help=f'Default: {OUTDIR}')
    parser.add_argument('--keep_bad',
                        action='store_true',
                        default=False,
                        help='Default: False')
    parser.add_argument('--dropna_th',
                        type=float,
                        default=0,
                        help='Default: 0')
    parser.add_argument('--src',
                        nargs='+',
                        default=None,
                        choices=['ccle', 'gcsi', 'gdsc', 'ctrp', 'nci60'],
                        help='Data sources to extract (default: None).')

    args = parser.parse_args(args)
    return args


def run(args):
    t0 = time()

    # Response columns
    rsp_cols = ['AUC', 'AUC1', 'EC50', 'EC50se', 'R2fit', 'Einf', 'IC50', 'HS', 'AAC1', 'DSS1']

    # Analysis of fibro samples are implemented in ccle_fibroblast.py and ccle_preproc.R
    # fibro_names = ['CCLE.HS229T', 'CCLE.HS739T', 'CCLE.HS840T', 'CCLE.HS895T', 'CCLE.RKN',
    #                'CTRP.Hs-895-T', 'CTRP.RKN', 'GDSC.RKN', 'gCSI.RKN']

    # -----------------------------------------------
    #     Create outdir
    # -----------------------------------------------
    outdir = create_outdir(args['gout'], args)
    args['outdir'] = str(outdir)

    # -----------------------------------------------
    #     Logger
    # -----------------------------------------------
    lg = Logger( outdir/'gen.df.log' )
    print_fn = get_print_func( lg.logger )
    print_fn(f'File path: {filepath}')
    print_fn(f'\n{pformat(args)}')
    dump_dict(args, outpath=outdir/'gen.df.args')

    # -----------------------------------------------
    #     Load response data, and features
    # -----------------------------------------------
    rsp = load_rsp(RSP_FILENAME, src=args['src'], keep_bad=args['keep_bad'],
                   print_fn=print_fn)
    ge = load_ge(DATADIR, ge_norm=args['ge_norm'], print_fn=print_fn,
                 float_type=prfx_dtypes['ge'])
    dd = load_dd(DSC_FILENAME, dropna_th=args['dropna_th'], print_fn=print_fn,
                 float_type=prfx_dtypes['dd'], src=args['src'], outdir=outdir)

    # -----------------------------------------------
    #     Load cell and drug meta
    # -----------------------------------------------
    cmeta = pd.read_csv(DATADIR/CELL_META_FILENAME, sep='\t', header=None,
                        names=['CELL', 'CANCER_TYPE'])
    # cmeta = pd.read_csv(DATADIR/'combined_metadata_2018May.txt', sep='\t').rename(columns={'sample_name': 'CELL', 'core_str': 'CELL_CORE'})
    # cmeta = cmeta[['CELL', 'CELL_CORE', 'tumor_type_from_data_src']]

    dmeta = pd.read_csv(DATADIR/DRUG_META_FILENAME, sep='\t')
    dmeta.rename(columns={'ID': 'DRUG', 'NAME': 'DRUG_NAME',
                          'CLEAN_NAME': 'DRUG_CLEAN_NAME'}, inplace=True)
    # TODO: What's going on with CTRP and GDSC? Why counts are not consistent across the fields??
    # dmeta.insert(loc=1, column='SOURCE', value=dmeta['DRUG'].map(lambda x: x.split('.')[0].lower()))
    # print(dmeta.groupby('SOURCE').agg({'DRUG': 'nunique', 'DRUG_NAME': 'nunique', 'DRUG_CLEAN_NAME': 'nunique', 'PUBCHEM': 'nunique'}).reset_index())
    # print(dmeta.groupby('SOURCE').agg({'DRUG': 'nunique', 'DRUG_NAME': 'unique', 'DRUG_CLEAN_NAME': 'unique'}).reset_index())

    # -----------------------------------------------
    #     Merge data
    # -----------------------------------------------
    print_fn('\n{}'.format('-' * 40))
    print_fn('... Start merging response with other dataframes ...')
    print_fn('-' * 40)

    # Merge with cmeta
    print_fn('\nMerge response (rsp) and cell metadata (cmeta) ...')
    data = pd.merge(rsp, cmeta, on='CELL', how='left') # left join to keep all rsp values
    print_fn(f'data.shape  {data.shape}\n')
    print_fn(data.groupby('SOURCE').agg({'CELL': 'nunique'}).reset_index())
    del rsp

    # Merge with dmeta
    print_fn('\nMerge with drug metadata (dmeta) ...')
    data = pd.merge(data, dmeta, on='DRUG', how='left') # left join to keep all rsp values
    print_fn(f'data.shape  {data.shape}\n')
    print_fn(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())

    # Merge with ge
    print_fn('\nMerge with expression (ge) ...')
    data = pd.merge(data, ge, on='CELL', how='inner') # inner join to keep samples that have ge
    print_fn(f'data.shape {data.shape}\n')
    print_fn(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())
    del ge

    # Merge with dd
    print_fn('\nMerge with descriptors (dd) ...')
    data = pd.merge(data, dd, on='DRUG', how='inner') # inner join to keep samples that have dd
    print_fn(f'data.shape {data.shape}\n')
    print_fn(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())
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


