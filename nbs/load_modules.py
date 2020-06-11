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

from fit import *
import rpy2.robjects as robjects # TODO not sure we need this

# from nls_lm import *
# import rpy2.robjects as robjects # TODO not sure we need this

filepath = Path(os.getcwd())
print(filepath)