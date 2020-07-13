import argparse
import os
import sys
from time import time
from pathlib import Path
from pprint import pprint
import numpy as np
from sklearn.model_selection import train_test_split

# Great blog about keras-tuner
# www.curiousily.com/posts/hackers-guide-to-hyperparameter-tuning/
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch, BayesianOptimization, Hyperband
from kerastuner.engine.hypermodel import HyperModel
from kerastuner.engine.hyperparameters import HyperParameters

fpath = Path(__file__).resolve().parent
sys.path.append( str(fpath/'../src') )  # must convert to str

from utils.classlogger import Logger
from utils.utils import load_data, dump_dict, get_print_func
from ml.scale import scale_fea
from ml.data import extract_subset_fea
from datasplit.splitter import data_splitter, gen_single_split

t0 = time()
MAX_TRIALS = 50
EXECUTIONS_PER_TRIAL = 1
EPOCHS = 15
OBJECTIVE = 'val_mae'


def get_data(datapath, seed=None, tr_sz=None, vl_sz=None, data_prep_def=None,
             print_fn=print):
    data = load_data( datapath )
    print_fn('data.shape {}'.format(data.shape))
    trg_name = 'AUC'

    # Get features (x), target (y), and meta
    fea_list = ['ge', 'dd']
    fea_sep = '_'
    xdata = extract_subset_fea(data, fea_list=fea_list, fea_sep=fea_sep)
    meta = data.drop( columns=xdata.columns )
    ydata = meta[[ trg_name ]]
    del data

    xdata = scale_fea(xdata=xdata, scaler_name='stnd')
    tr_id, vl_id = train_test_split(range(xdata.shape[0]), test_size=vl_sz,
                                    random_state=seed)

    # kwargs = {'data': xdata,
    #           'te_method': 'simple',
    #           'cv_method': 'simple',
    #           'te_size': 20000,  # 0.1
    #           'mltype': 'reg',
    #           'ydata': ydata,
    #           'split_on': None,
    #           'seed': seed,
    #           'print_fn': print_fn}

    # print_fn('\nGenerate data splits ...')
    # # tr_dct, vl_dct, te_dct = data_splitter( n_splits=1, **kwargs )
    # # tr_id = tr_dct[0]
    # # vl_id = vl_dct[0]
    # # te_id = te_dct[0]
    # tr_id, vl_id, te_id = gen_single_split( **kwargs )

    xtr = xdata.iloc[tr_id, :].reset_index(drop=True)
    ytr = np.squeeze(ydata.iloc[tr_id, :]).reset_index(drop=True)

    xvl = xdata.iloc[vl_id, :].reset_index(drop=True)
    yvl = np.squeeze(ydata.iloc[vl_id, :]).reset_index(drop=True)

    if tr_sz is not None:
        xtr = xtr.iloc[:tr_sz, :]
        ytr = ytr.iloc[:tr_sz]

    if (vl_sz is not None) and (vl_sz < xvl.shape[0]):
        xvl = xvl.iloc[:vl_sz, :]
        yvl = yvl.iloc[:vl_sz]

    # xtr = xtr.values
    # ytr = ytr.values
    # xvl = xvl.values
    # yvl = yvl.values

    print_fn('xtr {}'.format(xtr.shape))
    print_fn('xvl {}'.format(xvl.shape))
    print_fn('ytr {}'.format(ytr.shape))
    print_fn('yvl {}'.format(yvl.shape))

    if data_prep_def is not None:
        xtr = data_prep_def(xtr)
        xvl = data_prep_def(xvl)
    else:
        xtr = np.asarray(xtr)
        xvl = np.asarray(xvl)

    ytr = np.asarray(ytr)
    yvl = np.asarray(yvl)

    del xdata
    return xtr, ytr, xvl, yvl


# ------------------------------------------------------------
def data_prep_nn0_def(xdata):
    """
    This func prepare the dataset for keras model.
    This function works in a similar way as DataLoader
    in PyTorch.
    """
    xdata = np.asarray( xdata )
    x_dct = {'inputs': xdata}
    return x_dct


def data_prep_nn1_def(xdata):
    """
    This func prepare the dataset for keras model.
    This function works in a similar way as DataLoader
    in PyTorch.
    """
    x_ge = extract_subset_fea(xdata, fea_list=['ge'], fea_sep='_')
    x_dd = extract_subset_fea(xdata, fea_list=['dd'], fea_sep='_')
    x_ge = np.asarray( x_ge )
    x_dd = np.asarray( x_dd )
    x_dct = {'in_ge': x_ge, 'in_dd': x_dd}
    return x_dct
# ------------------------------------------------------------


data_sources = ['gdsc', 'ctrp', 'nci60', 'top21']

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='gdsc', choices=data_sources)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--tr_sz', type=int, default=None)
parser.add_argument('--vl_sz', type=int, default=None)
parser.add_argument('--ml', type=str, default='nn_reg0',
                    choices=['nn_reg0', 'nn_reg1'])
args = parser.parse_args()

DATADIR = fpath/'../data/ml.dfs'
datapath = DATADIR/f'data.{args.source}.dd.ge.raw/data.{args.source}.dd.ge.raw.parquet'

print_fn = print
source = [str(s) for s in data_sources if s in str(datapath)][0]
if source is None:
    outdir = fpath/'out'
else:
    outdir = fpath/f'{source}_{args.ml}_tuner_out'
os.makedirs(outdir, exist_ok=True)

lg = Logger( outdir/'tuner.log' )
print_fn = get_print_func( lg.logger )

if args.ml == 'nn_reg0':
    data_prep_def = data_prep_nn0_def
elif args.ml == 'nn_reg1':
    data_prep_def = data_prep_nn1_def

import pdb; pdb.set_trace()
xtr, ytr, xvl, yvl = get_data(datapath,
                              seed=args.seed,
                              tr_sz=args.tr_sz,
                              vl_sz=args.vl_sz,
                              data_prep_def=data_prep_def,
                              print_fn=print)
tr_sz = len(ytr)


"""Basic case:
- We define a `build_model` function
- It returns a compiled model
- It uses hyperparameters defined on the fly
"""
# ------------------------------------------------------------


class HyModelNN0(HyperModel):

    def __init__(self, input_dim: int, initializer='he_uniform'):
        self.input_dim = input_dim
        self.initializer = initializer

    def build(self, hp):
    # def build_model():
        input_dim = self.input_dim
        initializer = self.initializer

        inputs = layers.Input(shape=(input_dim,), name='inputs')

        # HPs
        hp_dr_rate = hp.Float('dr_rate', min_value=0.0, max_value=0.4,
                              default=0.20, step=0.05)

        # batchnorm = hp.Boolean('batchnorm', default=False)
        batchnorm = True

        units = [1000, 1000, 500, 250, 125] # fair
        x = layers.Dense(units[0], kernel_initializer=initializer)(inputs)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(hp_dr_rate)(x)

        x = layers.Dense(units[1], kernel_initializer=initializer)(x)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(hp_dr_rate)(x)

        x = layers.Dense(units[2], kernel_initializer=initializer)(x)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(hp_dr_rate)(x)

        x = layers.Dense(units[3], kernel_initializer=initializer)(x)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(hp_dr_rate)(x)

        x = layers.Dense(units[4], kernel_initializer=initializer)(x)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(hp_dr_rate)(x)

        outputs = layers.Dense(1, activation='relu', name='outputs')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        hp_lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2,
                         sampling='LOG', default=1e-3)
        opt = keras.optimizers.Adam( hp_lr )
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
        return model


# ----------------------------------------------------------------


class HyModelNN1(HyperModel):

    def __init__(self, in_dim_ge: int, in_dim_dd: int, initializer='he_uniform'):
        self.in_dim_ge = in_dim_ge
        self.in_dim_dd = in_dim_dd
        self.initializer = initializer

    def build(self, hp):
    # def build_model(in_dim_ge, in_dim_dd):
        # initializer = 'he_uniform'
        # hp_dr_rate = 0.2
        in_dim_ge = self.in_dim_ge
        in_dim_dd = self.in_dim_dd
        initializer = self.initializer

        # HPs
        hp_dr_rate = hp.Float('dr_rate', min_value=0.0, max_value=0.4,
                              default=0.20, step=0.05)

        batchnorm = True

        # ---------------------
        # GE
        in_ge = layers.Input(shape=(in_dim_ge,), name='in_ge')
        units_ge = [800, 500]  # fair

        x = layers.Dense(units_ge[0], kernel_initializer=initializer, name='g_dense_0')(in_ge)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(hp_dr_rate)(x)

        x = layers.Dense(units_ge[1], kernel_initializer=initializer, name='g_dense_1')(x)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        out_ge = layers.Dropout(hp_dr_rate)(x)

        ge = keras.Model(inputs=in_ge, outputs=out_ge, name=f'out_ge')

        # ---------------------
        # DD
        in_dd = layers.Input(shape=(in_dim_dd,), name='in_dd')
        units_dd = [1000, 700]  # fair

        x = layers.Dense(units_dd[0], kernel_initializer=initializer, name='d_dense_0')(in_dd)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(hp_dr_rate)(x)

        x = layers.Dense(units_dd[1], kernel_initializer=initializer, name='d_dense_1')(x)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        out_dd = layers.Dropout(hp_dr_rate)(x)

        dd = keras.Model(inputs=in_dd, outputs=out_dd, name=f'out_dd')

        # ---------------------
        # Merge towers
        mrg = layers.concatenate([ge.output, dd.output], axis=1)
        units_mrg = [500, 250, 125]  # fair

        x = layers.Dense(units_mrg[0], kernel_initializer=initializer)(mrg)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(hp_dr_rate)(x)

        x = layers.Dense(units_mrg[1], kernel_initializer=initializer)(x)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(hp_dr_rate)(x)

        x = layers.Dense(units_mrg[2], kernel_initializer=initializer)(x)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(hp_dr_rate)(x)

        # ---------------------
        # Output
        outputs = layers.Dense(1, activation='relu', name='outputs')(x)

        # ---------------------
        # Input --> Output
        model = keras.Model(inputs=[in_ge, in_dd], outputs=[outputs])

        # opt = keras.optimizers.Adam(learning_rate=0.0001)
        hp_lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2,
                         sampling='LOG', default=1e-3)
        opt = keras.optimizers.Adam( hp_lr )
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
        return model


# ------------------------------------------------------------
# model = build_model(in_dim_ge=xtr['in_ge'].shape[1],
#                     in_dim_dd=xtr['in_dd'].shape[1])
# model.fit(xtr, ytr, epochs=10, batch_size=32, verbose=1)

if args.ml == 'nn_reg0':
    hypermodel = HyModelNN0(input_dim=xtr['inputs'].shape[1])
elif args.ml == 'nn_reg1':
    hypermodel = HyModelNN1(in_dim_ge=xtr['in_ge'].shape[1],
                            in_dim_dd=xtr['in_dd'].shape[1])

# import pdb; pdb.set_trace()
# keras-team.github.io/keras-tuner/tutorials/distributed-tuning/
# dist_strategy = tf.distribute.MirroredStrategy()
dist_strategy = None

proj_name = f'tr_sz_{tr_sz}'
# tuner = BayesianOptimization(
# tuner = Hyperband(
tuner = RandomSearch(
    hypermodel,
    objective=OBJECTIVE,
    max_trials=MAX_TRIALS,
    executions_per_trial=EXECUTIONS_PER_TRIAL,
    distribution_strategy=dist_strategy,
    directory=outdir,
    project_name=proj_name,
    overwrite=True)

tuner.search_space_summary()


# import pdb; pdb.set_trace()
print_fn('Start HP search ...')
tuner.search(x=xtr, y=ytr,
             epochs=EPOCHS,
             batch_size=32,
             validation_data=(xvl, yvl))

my_log_out = outdir/proj_name/'my_logs'
os.makedirs(my_log_out, exist_ok=True)

tuner.results_summary()
model = tuner.get_best_models(num_models=1)[0]
results = model.evaluate(xvl, yvl, batch_size=128, verbose=0)
model.save( str(my_log_out/'best_model_trained' ))
print(results)

# Get dict with best HPs
best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
print('Attributes of a Trial object:\n')
pprint(best_trial.__dict__)
best_hps = best_trial.hyperparameters.values
best_hps.update({'tr_sz': tr_sz})
best_hps.update({'trial_id': best_trial.trial_id})
best_hps.update({OBJECTIVE: best_trial.score})
dump_dict(best_hps, outpath=my_log_out/'best_hps.txt')
print(best_hps)

# Return model with best HPs but untrained
best_hps = tuner.get_best_hyperparameters()[0]
model = tuner.hypermodel.build(best_hps)
results = model.evaluate(xvl, yvl, batch_size=128, verbose=0)
model.save( str(my_log_out/'best_model_untrained' ))
print(results)
# best_model.fit(...)

print('Runtime: {:.1f} hrs'.format( (time()-t0)/3600 ))
