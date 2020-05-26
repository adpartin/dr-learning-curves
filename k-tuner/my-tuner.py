import os
import sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

fpath = Path(__file__).resolve().parent
# sys.path.append( os.path.abspath(fpath/'../src/utils') )
# sys.path.append( str(fpath/'../src/utils') ) # must convert to str
sys.path.append( str(fpath/'../src') ) # must convert to str

from utils.utils import load_data, dump_dict, get_print_func
from ml.scale import scale_fea
from ml.data import extract_subset_fea
from datasplit.splitter import data_splitter, gen_single_split

from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hypermodel import HyperModel
from kerastuner.engine.hyperparameters import HyperParameters

parser = argparse.ArgumentParser()
parser.add_argument('--sr', type=str, default='ccle')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--sz', type=int, default=None)
args = parser.parse_args()


datapath = '/vol/ml/apartin/projects/dr-learning-curves/data/ml.data/data.ccle.dsc.rna.raw/data.ccle.dsc.rna.raw.parquet'
# datapath = '/vol/ml/apartin/projects/dr-learning-curves/data/ml.data/data.gdsc.dsc.rna.raw/data.gdsc.dsc.rna.raw.parquet'
print_fn = print
source = [str(s) for s in ['ccle', 'gdsc', 'ctrp'] if s in datapath][0]
if source is None:
    outdir = fpath/'out'
else:
    outdir = fpath/f'{source}_out'
os.makedirs(outdir, exist_ok=True)

def get_data(datapath, seed=None, tr_sz=None):
    data = load_data( datapath )
    print_fn('data.shape {}'.format(data.shape))
    trg_name = 'AUC'

    # Get features (x), target (y), and meta
    fea_list = ['GE', 'DD']
    fea_sep = '_'
    xdata = extract_subset_fea(data, fea_list=fea_list, fea_sep=fea_sep)
    meta = data.drop( columns=xdata.columns )
    ydata = meta[[ trg_name ]]
    del data

    xdata = scale_fea(xdata=xdata, scaler_name='stnd')    

    kwargs = {'data': xdata, 'te_method' :'simple', 'cv_method': 'simple',
              'te_size': 0.1, 'mltype': 'reg', 'ydata': ydata,
              'split_on': None, seed=seed, 'print_fn': print_fn}

    print_fn('\nGenerate data splits ...')
    # tr_dct, vl_dct, te_dct = data_splitter( n_splits=1, **kwargs )
    tr_dct, vl_dct, te_dct = gen_single_split( n_splits=1, **kwargs )

    # Get the indices for this split
    tr_id = tr_dct[0]
    vl_id = vl_dct[0]
    te_id = te_dct[0]

    xtr = xdata.iloc[tr_id, :].reset_index(drop=True)
    ytr = np.squeeze(ydata.iloc[tr_id, :]).reset_index(drop=True)

    xvl = xdata.iloc[vl_id, :].reset_index(drop=True)
    yvl = np.squeeze(ydata.iloc[vl_id, :]).reset_index(drop=True)

    # tr_sz = 5000
    if tr_sz is not None:
        xtr = xtr.iloc[:tr_sz,:]
        ytr = ytr.iloc[:tr_sz]
    return xtr, ytr, xvl, yvl

xtr, ytr, xvl, yvl = get_data(datapath)

"""Basic case:
- We define a `build_model` function
- It returns a compiled model
- It uses hyperparameters defined on the fly
"""


def build_model(hp):
# def build_model():
    initializer='he_uniform'
    batchnorm=False
    input_dim=3762

    model = keras.Sequential()
    inputs = keras.layers.Input(shape=(input_dim,), name='inputs')

    units = [1000, 1000, 500, 250, 125, 60, 30]
    x = layers.Dense(units[0], kernel_initializer=initializer)(inputs)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
        
    x = layers.Dense(units[1], kernel_initializer=initializer)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
        
    x = layers.Dense(units[2], kernel_initializer=initializer)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
        
    x = layers.Dense(units[3], kernel_initializer=initializer)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
        
    x = layers.Dense(units[4], kernel_initializer=initializer)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
        
    x = layers.Dense(units[5], kernel_initializer=initializer)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
        
    x = layers.Dense(units[6], kernel_initializer=initializer)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
        
    outputs = layers.Dense(1, activation='relu', name='outputs')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # opt = keras.optimizers.Adam(learning_rate=0.0001)
    opt = keras.optimizers.Adam(
            hp.Float('learning_rate', min_value=1e-4, max_value=1e-3, sampling='LOG', default=1e-4)
            )
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
    return model

# import pdb; pdb.set_trace()
# model = build_model()
# model.fit(xtr, ytr, epochs=10, batch_size=32, verbose=1)

tuner = RandomSearch(
    build_model,
    objective='val_mae',
    max_trials=10,
    executions_per_trial=1,
    directory=outdir,
    project_name='LC')

tuner.search_space_summary()

import pdb; pdb.set_trace()
tuner.search(x=xtr, y=ytr, epochs=5, batch_size=32,
             validation_data=(xvl,  yvl))

tuner.results_summary()
model = tuner.get_best_models(num_models=1)

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



