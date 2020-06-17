import os
import sys
from pathlib import Path
import argparse
from pprint import pprint

import numpy as np
import pandas as pd

fpath = Path(__file__).resolve().parent
sys.path.append( str(fpath/'../src') ) # must convert to str

from utils.utils import load_data, dump_dict, get_print_func
from ml.scale import scale_fea
from ml.data import extract_subset_fea
from datasplit.splitter import data_splitter, gen_single_split

# Great blog about keras-tuner
# www.curiousily.com/posts/hackers-guide-to-hyperparameter-tuning/
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch, BayesianOptimization, Hyperband
from kerastuner.engine.hypermodel import HyperModel
from kerastuner.engine.hyperparameters import HyperParameters

MAX_TRIALS = 50
EXECUTIONS_PER_TRIAL = 1
EPOCHS = 40
OBJECTIVE = 'val_mae'

def get_data(datapath, seed=None, tr_sz=None):
    data = load_data( datapath )
    print_fn('data.shape {}'.format(data.shape))
    trg_name = 'AUC'

    # Get features (x), target (y), and meta
    # fea_list = ['GE', 'DD']
    # fea_sep = '_'
    fea_list = ['ge', 'dd']
    fea_sep = '_'
    xdata = extract_subset_fea(data, fea_list=fea_list, fea_sep=fea_sep)
    meta = data.drop( columns=xdata.columns )
    ydata = meta[[ trg_name ]]
    del data

    xdata = scale_fea(xdata=xdata, scaler_name='stnd')    

    kwargs = {'data': xdata, 'te_method' :'simple', 'cv_method': 'simple',
              'te_size': 0.1, 'mltype': 'reg', 'ydata': ydata,
              'split_on': None, 'seed': seed, 'print_fn': print_fn}

    print_fn('\nGenerate data splits ...')
    # tr_dct, vl_dct, te_dct = data_splitter( n_splits=1, **kwargs )
    # tr_id = tr_dct[0]
    # vl_id = vl_dct[0]
    # te_id = te_dct[0]
    tr_id, vl_id, te_id = gen_single_split( **kwargs )

    xtr = xdata.iloc[tr_id, :].reset_index(drop=True)
    ytr = np.squeeze(ydata.iloc[tr_id, :]).reset_index(drop=True)

    xvl = xdata.iloc[vl_id, :].reset_index(drop=True)
    yvl = np.squeeze(ydata.iloc[vl_id, :]).reset_index(drop=True)

    if tr_sz is not None:
        xtr = xtr.iloc[:tr_sz,:]
        ytr = ytr.iloc[:tr_sz]

    xtr = xtr.values
    ytr = ytr.values
    xvl = xvl.values
    yvl = yvl.values
    del xdata
    return xtr, ytr, xvl, yvl


parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='gdsc')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--sz', type=int, default=None)
parser.add_argument('--ml', type=int, default='nn0', choices=['nn0', 'nn1'])
args = parser.parse_args()

# import pdb; pdb.set_trace()
DATADIR = fpath/'../data/ml.dfs'
datapath = DATADIR/f'data.{args.source}.dd.ge.raw/data.{args.source}.dd.ge.raw.parquet'

print_fn = print
source = [str(s) for s in ['gdsc', 'ctrp'] if s in str(datapath)][0]
if source is None:
    outdir = fpath/'out'
else:
    outdir = fpath/f'{source}_{args.ml}_tuner_out'
os.makedirs(outdir, exist_ok=True)

xtr, ytr, xvl, yvl = get_data(datapath, seed=args.seed, tr_sz=args.sz)
tr_sz = xtr.shape[0]

"""Basic case:
- We define a `build_model` function
- It returns a compiled model
- It uses hyperparameters defined on the fly
"""
# class MyHyperModel(HyperModel):

#     def __init__(self, input_dim=None, batchnorm=False, initializer='he_uniform'):
#         self.input_dim = input_dim
#         self.initializer = initializer

#     # def build_model(self, hp):
#     def build(self, hp):
#     # def build_model():
#         # initializer='he_uniform'
#         # batchnorm=False
#         # input_dim=3762
#         input_dim = self.input_dim
#         initializer = self.initializer

#         inputs = keras.layers.Input(shape=(input_dim,), name='inputs')

#         # HPs
#         hp_dr_rate = hp.Float('dr_rate', min_value=0.0, max_value=0.5,
#                                default=0.25, step=0.05)

#         batchnorm = hp.Boolean('batchnorm', default=False) 

#         units = [1000, 1000, 500, 250, 125, 60, 30]
#         x = layers.Dense(units[0], kernel_initializer=initializer)(inputs)
#         if batchnorm:
#             x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         x = layers.Dropout(hp_dr_rate)(x)        
            
#         x = layers.Dense(units[1], kernel_initializer=initializer)(x)
#         if batchnorm:
#             x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         x = layers.Dropout(hp_dr_rate)(x)        
            
#         x = layers.Dense(units[2], kernel_initializer=initializer)(x)
#         if batchnorm:
#             x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         x = layers.Dropout(hp_dr_rate)(x)        
            
#         x = layers.Dense(units[3], kernel_initializer=initializer)(x)
#         if batchnorm:
#             x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         x = layers.Dropout(hp_dr_rate)(x)        
            
#         x = layers.Dense(units[4], kernel_initializer=initializer)(x)
#         if batchnorm:
#             x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         x = layers.Dropout(hp_dr_rate)(x)        
            
#         x = layers.Dense(units[5], kernel_initializer=initializer)(x)
#         if batchnorm:
#             x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         x = layers.Dropout(hp_dr_rate)(x)        
            
#         x = layers.Dense(units[6], kernel_initializer=initializer)(x)
#         if batchnorm:
#             x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         x = layers.Dropout(hp_dr_rate)(x)        
            
#         outputs = layers.Dense(1, activation='relu', name='outputs')(x)
#         model = keras.Model(inputs=inputs, outputs=outputs)
        
#         # opt = keras.optimizers.Adam(learning_rate=0.0001)
#         hp_lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-3, sampling='LOG', default=1e-4)
#         # hp_lr = hp.Choice('learning_rate', [1e-4, 1e-3])
#         opt = keras.optimizers.Adam( hp_lr )
#         model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
#         return model

# ------------------------------------------------------------
class HyModelNN0(HyperModel):

    def __init__(self, input_dim=None, batchnorm=False, initializer='he_uniform'):
        self.input_dim = input_dim
        self.initializer = initializer

    # def build_model(self, hp):
    def build(self, hp):
    # def build_model():
        input_dim = self.input_dim
        initializer = self.initializer

        inputs = keras.layers.Input(shape=(input_dim,), name='inputs')

        # HPs
        hp_dr_rate = hp.Float('dr_rate', min_value=0.0, max_value=0.4,
                               default=0.20, step=0.05)

        # batchnorm = hp.Boolean('batchnorm', default=False) 
        batchnorm = True

        # units = [1000, 1000, 500, 250, 125, 60, 30] # original
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
        
        # opt = keras.optimizers.Adam(learning_rate=0.0001)
        hp_lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2,
                         sampling='LOG', default=1e-3)
        # hp_lr = hp.Choice('learning_rate', [1e-4, 1e-3])
        opt = keras.optimizers.Adam( hp_lr )
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
        return model
# ------------------------------------------------------------

# model = build_model()
# model.fit(xtr, ytr, epochs=10, batch_size=32, verbose=1)

## hypermodel = MyHyperModel( input_dim=xtr.shape[1] )
hypermodel = HyModelNN0( input_dim=xtr.shape[1] )
# hypermodel = HyModelNN1( input_dim=xtr.shape[1] )

# import pdb; pdb.set_trace()
# keras-team.github.io/keras-tuner/tutorials/distributed-tuning/
# dist_strategy = tf.distribute.MirroredStrategy()
dist_strategy = None

proj_name = f'tr_sz_{tr_sz}'
tuner = BayesianOptimization(
# tuner = Hyperband(
# tuner = RandomSearch(
    # build_model,
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
tuner.search(x=xtr, y=ytr, epochs=EPOCHS, batch_size=32,
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
# =================================================================

# # """Case #2:
# # - We override the loss and metrics
# # """
# tuner = RandomSearch(
#     build_model,
#     objective='val_accuracy',
#     loss=keras.losses.SparseCategoricalCrossentropy(name='my_loss'),
#     metrics=['accuracy', 'mse'],
#     max_trials=5,
#     directory='test_dir')

# tuner.search(x, y,
#              epochs=5,
#              validation_data=(val_x, val_y))

# # """Case #3:
# # - We define a HyperModel subclass
# # """
# class MyHyperModel(HyperModel):

#     def __init__(self, img_size, num_classes):
#         self.img_size = img_size
#         self.num_classes = num_classes

#     def build(self, hp):
#         model = keras.Sequential()
#         model.add(layers.Flatten(input_shape=self.img_size))
#         for i in range(hp.Int('num_layers', 2, 20)):
#             model.add(layers.Dense(units=hp.Int('units_' + str(i), 32, 512, 32),
#                                    activation='relu'))
#         model.add(layers.Dense(self.num_classes, activation='softmax'))
#         model.compile(
#             optimizer=keras.optimizers.Adam(
#                 hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
#             loss='sparse_categorical_crossentropy',
#             metrics=['accuracy'])
#         return model

# tuner = RandomSearch(
#     MyHyperModel(img_size=(28, 28), num_classes=10),
#     objective='val_accuracy',
#     max_trials=5,
#     directory='test_dir')

# tuner.search(x,
#              y=y,
#              epochs=5,
#              validation_data=(val_x, val_y))

# # """Case #4:
# # - We restrict the search space
# # - This means that default values are being used for params that are left out
# # """
# hp = HyperParameters()
# hp.Choice('learning_rate', [1e-1, 1e-3])

# tuner = RandomSearch(
#     build_model,
#     max_trials=5,
#     hyperparameters=hp,
#     tune_new_entries=False,
#     objective='val_accuracy')

# tuner.search(x=x,
#              y=y,
#              epochs=5,
#              validation_data=(val_x, val_y))

# # """Case #5:
# # - We override specific parameters with fixed values that aren't the default
# # """
# hp = HyperParameters()
# hp.Fixed('learning_rate', 0.1)

# tuner = RandomSearch(
#     build_model,
#     max_trials=5,
#     hyperparameters=hp,
#     tune_new_entries=True,
#     objective='val_accuracy')

# tuner.search(x=x,
#              y=y,
#              epochs=5,
#              validation_data=(val_x, val_y))

# # """Case #6:
# # - We reparameterize the search space
# # - This means that we override the distribution of specific hyperparameters
# # """
# hp = HyperParameters()
# hp.Choice('learning_rate', [1e-1, 1e-3])

# tuner = RandomSearch(
#     build_model,
#     max_trials=5,
#     hyperparameters=hp,
#     tune_new_entries=True,
#     objective='val_accuracy')

# tuner.search(x=x,
#              y=y,
#              epochs=5,
#              validation_data=(val_x, val_y))

# # """Case #7:
# # - We predefine the search space
# # - No unregistered parameters are allowed in `build`
# # """
# hp = HyperParameters()
# hp.Choice('learning_rate', [1e-1, 1e-3])
# hp.Int('num_layers', 2, 20)

# def build_model(hp):
#     model = keras.Sequential()
#     model.add(layers.Flatten(input_shape=(28, 28)))
#     for i in range(hp.get('num_layers')):
#         model.add(layers.Dense(32,
#                                activation='relu'))
#     model.add(layers.Dense(10, activation='softmax'))
#     model.compile(
#         optimizer=keras.optimizers.Adam(hp.get('learning_rate')),
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy'])
#     return model

# tuner = RandomSearch(
#     build_model,
#     max_trials=5,
#     hyperparameters=hp,
#     allow_new_entries=False,
#     objective='val_accuracy')

# tuner.search(x=x,
#              y=y,
#              epochs=5,
#              validation_data=(val_x, val_y))



