import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
from pprint import pprint, pformat

import numpy as np
import pandas as pd


def verify_path(path):
    """ Verify that the path exists. """
    if path is None:
        sys.exit('Program terminated. You must specify a correct path.')
    path = Path(path)
    assert path.exists(), f'The specified path was not found: {path}.'
    return path


def load_data(datapath, file_format=None):
    datapath = verify_path(datapath)
    if file_format is None:
        file_format = str(datapath).split('.')[-1]

    if file_format=='parquet':
        data = pd.read_parquet(datapath)
    elif file_format=='hdf5':
        data = pd.read_hdf5(datapath)
    elif file_format=='csv':
        data = pd.read_csv(datapath)
    else:
        try:
            data = pd.read_csv(datapath)
        except:
            print('Cannot load file', datapath)
    return data


def drop_dup_rows(data, print_fn=print):
    """ Drop duplicate rows. """
    print_fn('\nDrop duplicates ...')
    cnt0 = data.shape[0]; print_fn('Samples: {}'.format( cnt0 ))
    data = data.drop_duplicates().reset_index(drop=True)
    cnt1 = data.shape[0]; print_fn('Samples: {}'.format( cnt1 ));
    print_fn('Dropped duplicates: {}'.format( cnt0-cnt1 ))
    return data


def dropna(df, axis: int=0, th: float=0.05, max_na: int=None):
    """
    Drop rows (axis=0) or cols (axis=1) based on the ratio of NA values
    along the axis. Instead of ratio, you can also specify the max number
    of NA.
    Args:
        th (float): if ratio of NA values along the axis is larger that th,
                     then drop all the values
        max_na (int): specify max allowable number of na (instead of
                       specifying the ratio)
        axis (int): 0 to drop rows; 1 to drop cols
    """
    assert (axis in [0, 1]), "Invalid value for arg 'axis'."
    axis = 0 if (axis == 1) else 1

    if max_na is not None:
        assert max_na >= 0, 'max_na must be >=0.'
        idx = df.isna().sum(axis=axis) <= max_na
    else:
        idx = df.isna().sum(axis=axis)/df.shape[axis] <= th

    if axis == 0:
        df = df.iloc[:, idx.values]
    else:
        df = df.iloc[idx.values, :].reset_index(drop=True)
    return df


def dump_dict(dct, outpath='./dict.txt'):
    """ Dump dict into file. """
    with open(Path(outpath), 'w') as file:
        for k in sorted(dct.keys()):
            file.write('{}: {}\n'.format(k, dct[k]))


def get_print_func(logger=None):
    """ Returns the python 'print' function if logger is None. Othersiwe, returns logger.info. """
    return print if logger is None else logger.info


