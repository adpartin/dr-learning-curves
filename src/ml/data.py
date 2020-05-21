# import sklearn
import numpy as np
import pandas as pd


def extract_subset_fea(df, fea_list, fea_sep='_'):
    """ Extract features based feature prefix name. """
    fea = [c for c in df.columns if (c.split(fea_sep)[0]) in fea_list]
    return df[fea]
    
    
def extract_subset_fea_col_names(df, fea_list, fea_sep='_'):
    """ Extract features based feature prefix name. """
    return [c for c in df.columns if (c.split(fea_sep)[0]) in fea_list]    
    
            
def cnt_fea(df, fea_sep='_', verbose=True, print_fn=print):
    """ Count the number of features per feature type. """
    # print_fn = get_print_fn(logger)
    dct = {}
    unq_prfx = df.columns.map(lambda x: x.split(fea_sep)[0]).unique() # unique feature prefixes
    for prfx in unq_prfx:
        fea_type_cols = [c for c in df.columns if (c.split(fea_sep)[0]) in prfx] # all fea names of specific type
        dct[prfx] = len(fea_type_cols)
    if verbose:
        print_fn( pformat(dct) )
    return dct


