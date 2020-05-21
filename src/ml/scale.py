from pathlib import Path
import sklearn
import numpy as np
import pandas as pd

# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# from sklearn.externals import joblib
# from math import sqrt


def scale_fea(xdata, scaler_name='stnd', dtype=np.float32, verbose=False):
    """ Returns the scaled dataframe of features. """
    if scaler_name is None:
        if verbose:
            print('Scaler is None (not scaling).')
        return xdata
    
    if scaler_name == 'stnd':
        scaler = StandardScaler()
    elif scaler_name == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_name == 'rbst':
        scaler = RobustScaler()
    else:
        print(f'The specified scaler {scaler_name} is not supported (not scaling).')
        return xdata

    cols = xdata.columns
    return pd.DataFrame( scaler.fit_transform(xdata), columns=cols, dtype=dtype )    
    

