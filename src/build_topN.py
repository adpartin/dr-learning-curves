#!/usr/bin/env python

import os
import sys
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from functools import reduce
from pprint import pprint, pformat

# input files
base_data_dir = './data/raw'
response_path = Path('./data/raw/combined_single_response_agg')
cell_cancer_types_map_path = Path('./data/raw/combined_cancer_types')
drug_list_path = Path('./data/raw/drugs_1800')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

filepath = Path(__file__).resolve().parent  # (ap)

# (ap)  Utils
from utils.classlogger import Logger
from utils.utils import load_data, dump_dict, get_print_func
from utils.impute import impute_values
# from ml.scale import scale_fea
# from ml.data import extract_subset_fea


# Default settings
DATADIR = Path( filepath/'../data/raw' ).resolve()
OUTDIR = Path( DATADIR/'../ml.dfs' ).resolve()
os.makedirs(OUTDIR, exist_ok=True)


def parse_arguments(model_name=''):
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_n',
                        type=int, 
                        default=6,
                        help='Number of cancer types to be included. Default: 6')
    parser.add_argument('--drug_descriptor',
                        type=str,
                        default='dragon7',
                        choices=['dragon7', 'mordred'],
                        help='Drug descriptors. Default: dragon7')
    parser.add_argument('--cell_feature',
                        default='rnaseq',
                        choices=['rnaseq', 'snps'],
                        help='Cell line features. Default: rnaseq')
    parser.add_argument('--cell_feature_subset',
                        default='lincs1000',
                        choices=['lincs1000', 'oncogenes', 'all'],
                        help='Subset of cell line features. Default lincs1000')
    parser.add_argument('--format',
                        default='parquet',
                        choices=['csv', 'tsv', 'parquet', 'hdf5', 'feather'],
                        help='Dataframe file format. Default: parquet')
    parser.add_argument('--response_type',
                        default='reg',
                        choices=['reg', 'bin'],
                        help='Response type. Regression(reg) or Binary Classification(bin). Default reg')
    parser.add_argument('--labels',
                        action='store_true',
                        help='Contains Cell and Drug label. Default: False')
    parser.add_argument('--target', 
                        type=str,
                        default='AUC',
                        choices=['AUC', 'IC50', 'EC50', 'EC50se', 'R2fit', 'Einf', 'HS', 'AAC1',
                                 'AUC1', 'DSS1'],
                        help='Response label value. Default: AUC')
    parser.add_argument('--seed', 
                        type=int,
                        default=0,
                        help='Seed number (Default: 0)') # (ap)
    parser.add_argument('--src',
                        nargs='+',
                        default=None,
                        choices=['ccle', 'gcsi', 'gdsc', 'ctrp', 'nci60'],
                        help='Data sources to extract (default: None).') # (ap)

    args, unparsed = parser.parse_known_args()
    return args, unparsed


def check_file(filepath):
    print("checking {}".format(filepath))
    status = filepath.is_file()
    if status is False:
        print("File {} is not found in data dir.".format(filepath))
    return status


def check_data_files(args):
    filelist = [response_path, cell_cancer_types_map_path, drug_list_path,
                get_cell_feature_path(args), get_drug_descriptor_path(args)]
    return reduce((lambda x, y: x & y), map(check_file, filelist))


def get_cell_feature_path(args, rna_norm='combat'):
    if args.cell_feature_subset == 'all':
        filename = 'combined_{}_data_combat'.format(args.cell_feature)
    else:
        filename = 'combined_{}_data_{}_combat'.format(args.cell_feature, args.cell_feature_subset)
    return Path(base_data_dir, filename)


def get_drug_descriptor_path(args):
    filename = 'combined_{}_descriptors'.format(args.drug_descriptor)
    return Path(base_data_dir, filename)


def build_file_basename(args):
    return "top_{}.res_{}.cf_{}.dd_{}{}".format(
        args.top_n, args.response_type, args.cell_feature, args.drug_descriptor,
                                                '.labled' if args.labels else '')


def build_filename(args):
    # return "{}.{}".format(build_file_basename(args), args.format) # (ap) commented
    # (ap) added
    if args.top_n > 200:
        src = '' if args.src is None else '_'.join(args.src)
        fname = "data.{}.res_{}.cf_{}.dd_{}{}".format(
                src, args.response_type, args.cell_feature, args.drug_descriptor,
                '.labled' if args.labels else '')
    else:
        fname = "data.top{}.res_{}.cf_{}.dd_{}{}".format(
            args.top_n, args.response_type, args.cell_feature, args.drug_descriptor,
            '.labled' if args.labels else '')
    return fname
        

def dropna(df, axis=0, th=0.4):
    """ Drop rows or cols based on the ratio of NA values along the axis.
    Args:
        th (float) : if the ratio of NA values along the axis is larger that th, then drop all the values
        axis (int) : 0 to drop rows; 1 to drop cols
    """
    df = df.copy()
    axis = 0 if axis==1 else 1
    col_idx = df.isna().sum(axis=axis)/df.shape[axis] <= th
    df = df.iloc[:, col_idx.values]
    return df        
        

def build_dataframe(args):
    na_values = ['na', '-', ''] # (ap)
    
    # (ap) Create outdir and logger
    # outdir = Path('top' + str(args.top_n) + sffx + '_data')
    sffx = '' if args.src is None else '_'.join(args.src)
    if args.top_n < 200:
        outdir = Path(OUTDIR, 'data.' + 'top' + str(args.top_n) + sffx)
    else:
        outdir = Path(OUTDIR, sffx)
    os.makedirs(outdir, exist_ok=True)    
    
    # -----------------------------------------------
    #     Logger
    # -----------------------------------------------
    lg = Logger( outdir/'gen.df.log' )
    print_fn = get_print_func( lg.logger )
    print_fn(f'File path: {filepath}')
    print_fn(f'\n{pformat(args)}')
    dump_dict(vars(args), outpath=outdir/'gen.df.args')

    # -----------------------------------------------
    #     Identify Top N cancer types
    # -----------------------------------------------
    rsp = pd.read_csv(response_path, sep='\t', engine='c',
                              low_memory=False, na_values=na_values, warn_bad_lines=True)
    rsp.drop(columns='STUDY', inplace=True) # gives error when saves in 'parquet' format
    print_fn(rsp.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index()) # (ap)
    
    # (ap) Extract specific data sources
    rsp['SOURCE'] = rsp['SOURCE'].apply(lambda x: x.lower()) # (ap)
    if args.src is not None:
        rsp = rsp[rsp['SOURCE'].isin(args.src)].reset_index(drop=True)

    df_uniq_cl_drugs = rsp[['CELL', 'DRUG']].drop_duplicates().reset_index(drop=True)

    df_cl_cancer_map = pd.read_csv(cell_cancer_types_map_path, sep='\t', header=None,
                                   names=['CELL', 'CANCER_TYPE'])
    df_cl_cancer_map.set_index('CELL')

    df_cl_cancer_drug = df_cl_cancer_map.merge(df_uniq_cl_drugs, on='CELL', how='left', sort='true')
    df_cl_cancer_drug['CELL_DRUG'] = df_cl_cancer_drug.CELL.astype(str) + '.' + df_cl_cancer_drug.DRUG.astype(str)

    top_n = df_cl_cancer_drug.groupby(['CANCER_TYPE']).count().sort_values('CELL_DRUG', ascending=False).head(args.top_n)
    top_n_cancer_types = top_n.index.to_list()

    print_fn("Identified {} cancer types: {}".format(args.top_n, top_n_cancer_types))

    # -----------------------------------------------
    # Indentify cell lines associated with the target cancer types
    # -----------------------------------------------
    df_cl = df_cl_cancer_drug[df_cl_cancer_drug['CANCER_TYPE'].isin(top_n_cancer_types)][['CELL']].drop_duplicates().reset_index(drop=True)

    # -----------------------------------------------
    # Identify drugs associated with the target cancer type & filtered by drug_list
    # -----------------------------------------------
    df_drugs = df_cl_cancer_drug[df_cl_cancer_drug['CANCER_TYPE'].isin(top_n_cancer_types)][['DRUG']].drop_duplicates().reset_index(drop=True)

    drug_list = pd.read_csv(drug_list_path)['DRUG'].to_list()
    df_drugs = df_drugs[df_drugs['DRUG'].isin(drug_list)].reset_index(drop=True)

    # -----------------------------------------------
    # Filter response by cell lines (4882) and drugs (1779)
    # -----------------------------------------------
    cl_filter = df_cl.CELL.to_list()
    dr_filter = df_drugs.DRUG.to_list()
    target = args.target

    idx = rsp.CELL.isin(cl_filter) & rsp.DRUG.isin(dr_filter)
    rsp = rsp[idx].drop_duplicates().reset_index(drop=True) # (ap) keep all targets

    # (ap) Drop points with bad fit
    print_fn('\nDrop samples with bad fit (R2fit) ...')
    print_fn(f'rsp.shape {rsp.shape}')
    id_drop = rsp['R2fit'] <= 0
    rsp = rsp.loc[~id_drop,:]
    print_fn(f'Dropped {sum(id_drop)} rsp data points.')
    print_fn(f'rsp.shape {rsp.shape}')
    
    if args.response_type == 'bin':
        rsp[target] = rsp[target].apply(lambda x: 0 if x < 0.5 else 1)
        rsp.rename(columns={target: 'Response'}, inplace=True)

    # ----------------
    # Load RNA-Seq
    # ----------------        
    # Join response data with Drug descriptor & RNASeq
    ge = pd.read_csv(get_cell_feature_path(args), sep='\t', low_memory=False,
                     na_values=na_values, warn_bad_lines=True)
    ge = ge.astype(dtype={c: np.float32 for c in ge.columns[1:]})  # Cast features
    ge = ge[ge['Sample'].isin(cl_filter)].reset_index(drop=True)

    ge.rename(columns={'Sample': 'CELL'}, inplace=True)
    ge.columns = ['ge_' + x if i > 0 else x for i, x in enumerate(ge.columns.to_list())]
    ge = ge.set_index(['CELL'])

    # ----------------
    # Load descriptors
    # ----------------
    # dd_path = get_drug_descriptor_path(args) # original
    dd_path = Path(base_data_dir, 'pan_drugs_dragon7_descriptors.tsv') # (ap)
    dd = pd.read_csv(dd_path, sep='\t', low_memory=False,
                     na_values=na_values, warn_bad_lines=True)
    dd = dd.astype(dtype={c: np.float32 for c in dd.columns[1:]})  # Cast features
    dd.rename(columns={'NAME': 'DRUG'}, inplace=True) # (ap)
    dd = dd.rename(columns={c: 'dd_'+c for c in dd.columns[1:]}) # prefix drug desc names
    
    # (ap) Some features have too many NA values (drop these)
    print_fn('\nDrop cols with too many NA values ...')
    print_fn(f'dd.shape {dd.shape}')
    dropna_th = 0.05
    dd = dropna(dd, axis=1, th=dropna_th)
    print_fn(f'dd.shape {dd.shape}')    
    
    # (ap) There are descriptors where there is a single unique value excluding NA (drop those)
    print_fn('Drop descriptors that have a single unique value (excluding NAs) ...')
    col_idx = dd.nunique(dropna=True).values==1
    dd = dd.iloc[:, ~col_idx]
    print_fn(f'dd.shape {dd.shape}')

    # (ap) Impute missing values (drug descriptors)
    print_fn('Impute NA values ...')
    dd = impute_values(data=dd, print_fn=print_fn)

    # (ap)
    # There are still lots of descriptors which have only a few unique values.
    # We can categorize those values. e.g.: 564 descriptors have only 2 unique vals,
    # and 154 descriptors have only 3 unique vals, etc.
    # todo: use utility code from p1h_alex/utils/data_preproc.py that transform those
    # features into categorical and also applies an appropriate imputation.
    # dd.nunique(dropna=True).value_counts()[:10]
    # dd.nunique(dropna=True).value_counts().sort_index()[:10]
    
    # dd = dd[dd.DRUG.isin(dr_filter)].set_index(['DRUG']).fillna(0) # (ap) commented --> bad imputation!
    dd = dd[dd.DRUG.isin(dr_filter)].set_index(['DRUG']) # (ap) added --> drop data imputation!

    # -----------------------------------------------
    #     Merge data
    # -----------------------------------------------
    print_fn('\n{}'.format('-' * 40))
    print_fn('... Start merging response with other dataframes ...')
    print_fn('-' * 40)

    df = rsp.merge(ge, on='CELL', how='left', sort='true')
    df.set_index(['DRUG']) # TODO: this doesn't take effect unless performed 'inplace'
    
    df_final = df.merge(dd, on='DRUG', how='left', sort='true')
    if args.labels:
        df_cell_map = df_final['CELL'].to_dict()
        df_drug_map = df_final['DRUG'].to_dict()
        df_final.drop(columns=['CELL', 'DRUG'], inplace=True)
        df_final.drop_duplicates(inplace=True)
        df_final.insert(0, 'DRUG', df_final.index.map(df_drug_map))
        df_final.insert(0, 'CELL', df_final.index.map(df_cell_map))
        df_final.reset_index(drop=True, inplace=True)
    else:
        df_final.drop(columns=['CELL', 'DRUG'], inplace=True)
        df_final.drop_duplicates(inplace=True)
    print_fn("\nDataframe is built with total {} rows.".format(len(df_final)))

    # (ap) Shuffle
    # print_fn("Shuffle final df.")
    # df_final = df_final.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    print_fn(df_final.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index()) # (ap)
    tmp = df_final.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index()
    print_fn( tmp.iloc[:, 1:].sum(axis=0) )
    
    save_filename = build_filename(args)
    save_filename = outdir/save_filename
    
    if args.format == 'feather':
        df_final.to_feather(save_filename)
    elif args.format == 'csv':
        df_final.to_csv(str(save_filename) + '.csv', float_format='%g', index=False)
    elif args.format == 'tsv':
        df_final.to_csv(save_filename, sep='\t', float_format='%g', index=False)
    elif args.format == 'parquet':
        df_final.to_parquet(str(save_filename) + '.parquet', index=False)
    elif args.format == 'hdf5':
        df_cl.to_csv(build_file_basename(args) + '_cellline.txt', header=False, index=False)
        df_drugs.to_csv(build_file_basename(args) + '_drug.txt', header=False, index=False)
        df_final.to_hdf(save_filename, key='df', mode='w', complib='blosc:snappy', complevel=9)

    # Memory usage
    print_fn('\nTidy dataframe: {:.2f} GB'.format(sys.getsizeof(df_final)/1e9))
    
    # --------------------------------------------------
    # (ap) tissue type histogram
    # --------------------------------------------------
    def plot_tissue_hist(top_n):
        aa = df_cl_cancer_drug[['CELL', 'DRUG', 'CANCER_TYPE']].merge(
            df_final[['CELL', 'DRUG', 'AUC']], on=['CELL', 'DRUG'], how='inner')
        aa = pd.DataFrame(aa['CANCER_TYPE'].value_counts())
        aa = aa.reset_index().rename(columns={'index': 'ctype', 'CANCER_TYPE': 'count'})
        aa['ctype'] = aa['ctype'].map(lambda x: ' '.join(x.split('_')))

        x = aa['ctype']
        y = aa['count']
        ax = aa.plot.barh(x='ctype', y='count', xlim=[0, y.max()*1.15], legend=False, figsize=(9, 7), fontsize=12)
        ax.set_ylabel(None, fontsize=14);
        ax.set_xlabel('Total responses', fontsize=14);
        ax.set_title('Number of AUC responses per cancer type ({})'.format(top_n), fontsize=14);
        ax.invert_yaxis()

        for p in ax.patches:
            val = int(p.get_width()/1000)
            x = p.get_x() + p.get_width() + 1000
            y = p.get_y() + p.get_height()/2
            ax.annotate(str(val) + 'k', (x, y), fontsize=10)

        # OR
        # fig, ax = plt.subplots(figsize=(7, 5))
        # plt.barh(aa['CANCER_TYPE'], aa['CELL_DRUG'], color='b', align='center', alpha=0.7)
        # plt.xlabel('Total responses', fontsize=14);
        plt.savefig(outdir/'Top{}_histogram.png'.format(top_n), dpi=100, bbox_inches='tight')
        
        return aa
    
    aa = plot_tissue_hist(top_n=args.top_n)
    # --------------------------------------------------
    

    # --------------------------------------------------
    # (ap) break data
    # --------------------------------------------------
    # Split features and traget
    # print('\nSplit features and target.')
    
    # meta = df_final[['AUC', 'CELL', 'DRUG']]
    # xdata = df_final.drop(columns=['AUC', 'CELL', 'DRUG'])

    # xdata.to_parquet( outdir/'xdata.parquet' )
    # meta.to_parquet( outdir/'meta.parquet' )
    
    # print('Totoal DD: {}'.format( len([c for c in xdata.columns if 'DD' in c]) ))
    # print('Totoal GE: {}'.format( len([c for c in xdata.columns if 'GE' in c]) ))
    # --------------------------------------------------
    
    # --------------------------------------------------
    # (ap) generate train/val/test splits
    # --------------------------------------------------
    # from data_split import make_split
    # print('\nSplit train/val/test.')
    # args['cell_fea'] = 'GE'
    # args['drug_fea'] = 'DD'
    # args['te_method'] = 'simple'
    # args['cv_method'] = 'simple'
    # args['te_size'] = 0.1
    # args['vl_size'] = 0.1
    # args['n_jobs'] = 4
    # make_split(xdata=xdata, meta=meta, outdir=outdir, args=args)
    # --------------------------------------------------
    lg.kill_logger()
    print('Done.')


if __name__ == '__main__':
    FLAGS, unparsed = parse_arguments()
    if check_data_files(FLAGS):
        build_dataframe(FLAGS)
