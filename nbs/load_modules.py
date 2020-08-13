import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path

import sklearn
import numpy as np
import pandas as pd
from glob import glob

from sklearn import metrics
from math import sqrt

import matplotlib
import matplotlib.pyplot as plt
# plt.style.use('ggplot')

# Make all python scripts available in the path
# sys.path.append('../')

import pp_utils
import lrn_crv_plot

# sys.path.append('../')
fpath = Path(__file__).resolve().parent
sys.path.append( str(fpath/'../src') ) # must convert to str
from learningcurve import lc_plots

from fit import fit_params, biased_powerlaw
# import rpy2.robjects as robjects # TODO not sure we need this

from nls_lm import fit_model #, fit_params, biased_powerlaw

# filepath = Path(os.getcwd())
# print(filepath)


def drop_bad_r2(df):
    """ Remove runs with negative R2. """
    if df is None:
        return None
    else:
        print(df.shape)
        df['id'] = [str(sz)+'_'+str(r) for sz, r in zip(df['tr_size'], df['run'])]
        aa = df[ df['metric']=='r2' ]
        aa['valid'] = aa['score'] > 0
        ids_valid = aa[ aa['valid']==True ].id.unique()
        aa = df[ df['id'].isin(ids_valid) ]
        print(aa.shape)
    return aa


def print_count(aa):
    # aa = nn0_rnd
    aa['one'] = 1
    aa = aa[(aa['set']=='te') & (aa['metric']=='mean_absolute_error')]
    display(aa.groupby(['tr_size']).agg({'one': 'sum'}).sort_values('tr_size').reset_index())

    
def load_data(path, tr_set='te'):
    """ Load scores. """
    if path is None:
        return None
    else:
        df = pd.read_csv(path);
        df.rename(columns={'split': 'run'}, inplace=True)
        if tr_set != 'all':
            df = df[ df['set'] == tr_set ].reset_index(drop=True)
        return df


def load_data_hpc(path, tr_set='te'):
    """ Load scores from the runs on Summit HPC. """
    df = pd.read_csv(path);
    df.rename(columns={'split': 'run'}, inplace=True)
    if tr_set != 'all':
        df = df[ df['set'] == tr_set ].reset_index(drop=True)
    return df


def subset_data(df, col='tr_size', x_mn=None, x_mx=None):
    """ Subset df based on range. """
    if x_mn is not None:
        df = df[ df[col] >= x_mn ].reset_index(drop=True)
    if x_mx is not None:
        df = df[ df[col] <= x_mx ].reset_index(drop=True)
    return df


def add_weight_col(df, binomial=False):
    """ ... """
    if binomial:
        df['w'] = df['tr_size'] / ( df['y'] * (1-df['y']) )
    else:
        df['w'] = df['tr_size'] / df['tr_size'].max()
    return df


def pwr_law(x, a, b, c):
    y = a * x**(b) + c
    return y


def calc_fit(x, coefs):
    """ ... """
    coefs = coefs.reset_index(drop=True)
    args = { coefs.loc[i, 'coef']: coefs.loc[i, 'est'] for i in range(len(coefs)) }
    args.update({'x': x})
    y = pwr_law( **args )
    return y


# ------------------------------------------------------------------------
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


class FitPwrLaw():
    
    # def __init__(self, xf, yf, w=None):
    def __init__(self, xf, yf, w=None, a: float=1.2, b: float=-0.3, c: float=0.03):
        assert len(xf) == len(yf), 'xf and yf must be equal size.'
        self.xf = xf
        self.yf = yf
        if w is not None:
            assert len(xf) == len(w), 'xf and w must be equal size.'
            self.w = w
        else:
            w = np.ones( (len(self.xf),) )
            
        # New!
        self.a = a
        self.b = b
        self.c = c
            
        self.fit_model()
        # self.fit_params()

        
    def fit_model(self):
    # def fit_model(x: ro.IntVector, y: ro.FloatVector, w: ro.FloatVector):
        """ ... """
        x = ro.IntVector(list(self.xf))
        y = ro.FloatVector(list(self.yf))
        w = ro.FloatVector(list(self.w))

        # script = '\'../fit.R\''
        script = '\'nls_lm.R\''
        ro.r('''source({})'''.format(script))
        fit_nlsLM_power_law = ro.globalenv['fit_nlsLM_power_law']
        # coef_est_r = fit_nlsLM_power_law(x, y, w)  # commented
        coef_est_r = fit_nlsLM_power_law(x, y, w,
                                         a=self.a, b=self.b, c=self.c)  # new!

        # coef_est_py = pandas2ri.ri2py_dataframe(coef_est_r)
        with localconverter(ro.default_converter + pandas2ri.converter):
            coef_est_py = ro.conversion.rpy2py(coef_est_r)

        self.coefs = coef_est_py.reset_index(drop=True)
        self.a = self.coefs.loc[ self.coefs['coef'] == 'a', 'est'].values
        self.b = self.coefs.loc[ self.coefs['coef'] == 'b', 'est'].values
        self.c = self.coefs.loc[ self.coefs['coef'] == 'c', 'est'].values
    
    
    def fit_params(self):
        x = ro.IntVector(list(self.xf))
        y = ro.FloatVector(list(self.yf))
        
        script = '\'fit.R\''
        # script = '\'../fit.R\''
        ro.r('''source({})'''.format(script))
        get_params = ro.globalenv['model_param']
        a, b, c = get_params(x, y)

        # gamma, alpha, beta = get_params(x, y)
        prms = {}
        # prms_dct['alpha'], prms_dct['beta'], prms_dct['gamma'] = b, c-0.5, a

        prms['alpha'] = b
        prms['beta'] = c - 0.5
        prms['gamma'] = a
        
        self.prms = prms
        self.a = self.prms['alpha']
        self.b = self.prms['beta']
        self.c = self.prms['gamma']
        
    
    def calc_fit(self, x=None, x1=None, x2=None):
        """ Calculate the fit. """
        if x is not None:
            y = self.a * x**(self.b) + self.c
            
        elif (x1 is not None) and (x2 is not None):
            x = np.linspace(x1, x2, 50)
            y = self.a * x**(self.b) + self.c
            
        else:
            x = np.linspace(xf.min(), xf.max(), 50)
            y = self.a * x**(self.b) + self.c
            
        return x, y