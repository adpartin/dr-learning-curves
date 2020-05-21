import os
import sys
from pathlib import Path

# Utils
filepath = Path(__file__).resolve().parent
# sys.path.append( os.path.abspath(filepath/'../ml') )
from ml.keras_utils import r2_krs


try:
    import tensorflow as tf
    if int(tf.__version__.split('.')[0]) < 2:
        import keras
        from keras.models import load_model
        from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
        from keras.utils import plot_model
    else:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
        from tensorflow.keras.utils import plot_model

        from tensorflow.keras import backend as K
        from tensorflow.keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
        from tensorflow.keras import optimizers
        from tensorflow.keras.optimizers import SGD, Adam
        from tensorflow.keras.models import Sequential, Model
except:
    print('Could not import tensorflow.')


def clr_keras_callback(mode=None, base_lr=1e-4, max_lr=1e-3, gamma=0.999994):
    """ Creates keras callback for cyclical learning rate. """
    from . cyclical_learning_rate import CyclicLR
    if mode == 'trng1':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='triangular')
    elif mode == 'trng2':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='triangular2')
    elif mode == 'exp':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='exp_range', gamma=gamma) # 0.99994; 0.99999994; 0.999994
    return clr


# def r2(y_true, y_pred):
#     SS_res =  K.sum(K.square(y_true - y_pred))
#     SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
#     return (1 - SS_res/(SS_tot + K.epsilon()))


def reg_go_callback_def(outdir, ref_metric='val_loss', **clr_kwargs):
    """ Required for lrn_crv.py """
    checkpointer = ModelCheckpoint( str(outdir/'model_best.h5'), monitor='val_loss', verbose=0,
                                    save_weights_only=False, save_best_only=True )
    csv_logger = CSVLogger( outdir/'training.log' )
    reduce_lr = ReduceLROnPlateau( monitor=ref_metric, factor=0.75, patience=25, verbose=1,
                                   mode='auto', min_delta=0.0001, cooldown=3, min_lr=0.000000001 )
    early_stop = EarlyStopping( monitor=ref_metric, patience=50, verbose=1, mode='auto' )

    if bool(clr_kwargs):
        clr = clr_keras_callback( **clr_kwargs )
        return [checkpointer, csv_logger, early_stop, reduce_lr, clr]

    return [checkpointer, csv_logger, early_stop, reduce_lr]


def reg_go_arch(input_dim, dr_rate=0.1):
    DR = dr_rate
    inputs = Input(shape=(input_dim,))
    x = Dense(250, activation='relu')(inputs)
    x = Dropout(DR)(x)
    x = Dense(125, activation='relu')(x)
    x = Dropout(DR)(x)
    x = Dense(60, activation='relu')(x)
    x = Dropout(DR)(x)
    x = Dense(30, activation='relu')(x)
    x = Dropout(DR)(x)
    outputs = Dense(1, activation='relu')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def reg_go_model_def( **model_init ):
    model = reg_go_arch( **model_init )
    
    # www.tensorflow.org/addons/api_docs/python/tfa/optimizers/TriangularCyclicalLearningRate
    # from tensorflow.keras.optimizers import schedules
    # lr_schedule = schedules.TriangularCyclicalLearningRate(
    #     initial_learning_rate=1e-4,
    #     maximal_learning_rate=1e-3,
    #     step_size=2000,
    #     scale_mode='cycle',
    #     name='MyCyclicScheduler')
    # opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    
    opt = SGD(lr=0.0001, momentum=0.9)
    model.compile(loss='mean_squared_error',
                  optimizer=opt,
                  metrics=['mae', r2_krs])
    return model


