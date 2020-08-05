import os
import sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

filepath = Path(__file__).resolve().parent
from utils.utils import load_data
from ml.scale import scale_fea
from ml.data import extract_subset_fea


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--datapath', required=True, type=str, help='Full path to data.')
    parser.add_argument('--gout', type=str, default='.', help='Full path to data.')
    parser.add_argument('--n_jobs', default=8, type=int, help='Default: 8.')
    args = parser.parse_args(args)
    return args


def run(args):
    import pdb; pdb.set_trace()
    datapath = Path( args['datapath'] ).resolve()
    data = load_data( datapath )
    
    xdata = extract_subset_fea(data, fea_list=['ge','dd'], fea_sep='_')
    meta = data.drop( columns=xdata.columns )
    ydata = meta[['AUC']]
    del data
    
    xdata = scale_fea(xdata=xdata, scaler_name='stnd')    
    xtr, xvl, ytr, yvl = train_test_split(xdata, ydata, test_size=0.2)
    
    # -----------------------------------------------
    #      ML model configs
    # -----------------------------------------------
    from models.keras_model import nn_reg0_model_def, data_prep_nn0_def, model_callback_def
    keras_callbacks_def = model_callback_def
    data_prep_def = data_prep_nn0_def
    ml_model_def = nn_reg0_model_def

    ml_init_kwargs = {'input_dim': xdata.shape[1],
                      'dr_rate': 0.2,
                      'opt_name': 'adam', 
                      'lr': 0.001,
                      'batchnorm': True}
    ml_fit_kwargs = {'epochs': 2, 
                     'batch_size': 32,
                     'verbose': 1}

    xtr = data_prep_def( xtr )
    ytr = np.asarray( ytr )                
    xvl = data_prep_def( xvl )
    yvl = np.asarray( yvl ) 

    gout = Path(args['gout'])
    tmp_out = Path(gout, 'keras_callbask_out')
    os.makedirs(tmp_out, exist_ok=True)
    ml_fit_kwargs['validation_data'] = (xvl, yvl)
    ml_fit_kwargs['callbacks'] = keras_callbacks_def( outdir=tmp_out )

    model = ml_model_def(**ml_init_kwargs)
    # print(model.summary())

    history = model.fit(xtr, ytr, **ml_fit_kwargs)

    # -----------------------------------------------
    print('Done.')
    del xdata, ydata


def main(args):
    args = parse_args(args)
    args = vars(args)
    score = run(args)
    return score
    

if __name__ == '__main__':
    main(sys.argv[1:])


