"""
Functions to generate learning curves.
Records performance (error or score) vs training set size.
"""
import os
import sys
from pathlib import Path
from time import time
from glob import glob

import sklearn
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import metrics
from math import sqrt
from scipy import optimize

from pandas.api.types import is_string_dtype
from sklearn.preprocessing import LabelEncoder
import joblib

from datasplit.splitter import data_splitter
from ml.keras_utils import save_krs_history, plot_prfrm_metrics, r2_krs
from ml.evals import calc_preds, calc_scores, dump_preds
from utils.utils import dump_dict, verify_path
from utils.plots import plot_hist, plot_runtime


# --------------------------------------------------------------------------------
class LearningCurve():
    """
    Train estimator using multiple (train set) sizes and generate learning curves for multiple performance metrics.
    Example:
        lc = LearningCurve(xdata, ydata, cv_lists=(tr_ids, vl_ids))
        lc_scores = lc.trn_learning_curve(
            framework=framework, mltype=mltype, model_name=model_name,
            ml_init_args=ml_init_args, ml_fit_args=ml_fit_args, clr_keras_args=clr_keras_args)
    """
    def __init__(self,
            X, Y,
            meta=None,
            cv_lists=None,  # (tr_id, vl_id, te_id)
            n_splits: int=1,
            mltype: str='reg',
            lc_step_scale: str='log2',
            min_size = 0,
            max_size = None,
            lc_sizes: int=None,
            lc_sizes_arr: list=[],
            print_fn=print,
            save_model=False,
            outdir=Path('./')):
        """
        Args:
            X : array-like (pd.DataFrame or np.ndarray)
            Y : array-like (pd.DataFrame or np.ndarray)
            meta : array-like file of metadata (each item corresponds to an (x,y) sample
            cv : (optional) number of cv folds (int) or sklearn cv splitter --> scikit-learn.org/stable/glossary.html#term-cv-splitter
            cv_lists : tuple of 3 dicts, cv_lists[0] and cv_lists[1], cv_lists[2], that contain the tr, vl, and te folds, respectively
            cv_folds_arr : list that contains the specific folds in the cross-val run

            lc_step_scale : specifies how to generate the size values. 
                Available values: 'linear', 'log2', 'log10'.

            min_size : min size value in the case when lc_step_scale is 'log2' or 'log10'
            max_size : max size value in the case when lc_step_scale is 'log2' or 'log10'

            lc_sizes : number of sizes in the learning curve (used only in the lc_step_scale is 'linear')
            lc_sizes_arr : list of ints specifying the sizes to process (e.g., [128, 256, 512])
            
            size_frac : list of relative numbers of training samples that are used to generate learning curves
                e.g., size_frac=[0.1, 0.2, 0.4, 0.7, 1.0].
                If this arg is not provided, then the training sizes are generated from lc_sizes and lc_step_scale.
                
            save_model : dump model if True (keras model ignores this arg since we load the best model to calc score)
        """
        self.X = pd.DataFrame(X).reset_index(drop=True)
        self.Y = pd.DataFrame(Y).reset_index(drop=True)
        self.meta = pd.DataFrame(meta).reset_index(drop=True)

        self.cv_lists = cv_lists

        self.n_splits = n_splits
        self.mltype = mltype

        # self.cv_folds_arr = cv_folds_arr

        self.lc_step_scale = lc_step_scale 
        self.min_size = min_size
        self.max_size = max_size
        self.lc_sizes = lc_sizes
        self.lc_sizes_arr = lc_sizes_arr

        self.print_fn = print_fn
        self.save_model = save_model
        self.outdir = Path( outdir )

        self.create_split_dcts()
        self.create_tr_sizes_list()
        # self.trn_single_subset() # TODO: implement this method for better modularity


    def create_split_dcts(self):
        """
        Converts a tuple of arrays self.cv_lists into two dicts, tr_dct, vl_dct,
        and te_dict. Both sets of data structures contain the splits of all the
        k-folds.
        """
        tr_dct = {}
        vl_dct = {}
        te_dct = {}

        # Use lists passed as input arg
        if self.cv_lists is not None:
        #     tr_id = self.cv_lists[0]
        #     vl_id = self.cv_lists[1]
        #     te_id = self.cv_lists[2]
        #     assert (tr_id.shape[1]==vl_id.shape[1]) and (tr_id.shape[1]==te_id.shape[1]), 'tr, vl, and te must have the same number of folds.'
        #     self.cv_folds = tr_id.shape[1]

        #     # Calc the split ratio if cv=1
        #     # if self.cv_folds == 1:
        #     #     total_samples = tr_id.shape[0] + vl_id.shape[0] + te_id.shape[0]
        #     #     self.vl_size = vl_id.shape[0] / total_samples
        #     #     self.te_size = te_id.shape[0] / total_samples

        #     if self.cv_folds_arr is None:
        #         self.cv_folds_arr = [f+1 for f in range(self.cv_folds)]

        #     for fold in range(tr_id.shape[1]):
        #         # cv_folds_arr contains the specific folds we wish to process
        #         if fold+1 in self.cv_folds_arr:
        #             tr_dct[fold] = tr_id.iloc[:, fold].dropna().values.astype(int).tolist()
        #             vl_dct[fold] = vl_id.iloc[:, fold].dropna().values.astype(int).tolist()
        #             te_dct[fold] = te_id.iloc[:, fold].dropna().values.astype(int).tolist()

            tr_dct[0] = self.cv_lists[0]
            vl_dct[0] = self.cv_lists[1]
            te_dct[0] = self.cv_lists[2]

        # Generate splits on the fly if pre-computed splits were passed
        else:
            if isinstance(self.n_splits, int):
                if self.mltype=='cls':
                    te_method='strat'
                    cv_method='strat'
                else:
                    te_method='simple'
                    cv_method='simple'

                kwargs = {'data': self.X, 'te_method' :'simple', 'cv_method': 'simple',
                          'te_size': 0.1, 'mltype': self.mltype, 'ydata': self.Y,
                          'split_on': None, 'print_fn': print}

                self.print_fn('\nGenerate data splits ...')
                tr_dct, vl_dct, te_dct = data_splitter( n_splits=self.n_splits, **kwargs )
            else:
                raise ValueError(f'n_splits must be int>1. Got {n_splits}.')
            """
                # cv is sklearn splitter
                self.cv_folds = cv.get_n_splits()

            if cv_folds == 1:
                self.vl_size = cv.test_size

            # Create sklearn splitter 
            if self.mltype == 'cls':
                if self.Y.ndim > 1 and self.Y.shape[1] > 1:
                    splitter = self.cv.split(self.X, np.argmax(self.Y, axis=1))
            else:
                splitter = self.cv.split(self.X, self.Y)

            # Generate the splits
            for fold, (tr_vec, vl_vec) in enumerate(splitter):
                tr_dct[fold] = tr_vec
                vl_dct[fold] = vl_vec
            """
        # Keep dicts
        self.tr_dct = tr_dct
        self.vl_dct = vl_dct
        self.te_dct = te_dct


    def create_tr_sizes_list(self):
        """ Generate a list of training sizes (training sizes). """
        if self.lc_sizes_arr is not None:
            # No need to generate an array of training sizes if lc_sizes_arr is specified
            self.tr_sizes = np.asarray(self.lc_sizes_arr)

        else:
            # Fixed spacing
            key = list(self.tr_dct.keys())[0]
            total_trn_samples = len(self.tr_dct[key])
            if self.max_size is None:
                self.max_size = total_trn_samples # total number of available training samples
            else:
                if self.max_size > total_trn_samples:
                    self.max_size = total_trn_samples

            # Full vector of sizes
            # (we create a vector with very large values so that we later truncate it with max_size)
            scale = self.lc_step_scale.lower()
            if scale == 'linear':
                m = np.linspace(self.min_size, self.max_size, self.lc_sizes+1)[1:]
            else:
                # we create very large vector m, so that we later truncate it with max_size
                if scale == 'log2':
                    m = 2 ** np.array(np.arange(30))[1:]
                elif scale == 'log10':
                    m = 10 ** np.array(np.arange(8))[1:]
                elif scale == 'log':
                    if self.lc_sizes is not None:
                        # www.researchgate.net/post/is_the_logarithmic_spaced_vector_the_same_in_any_base
                        pw = np.linspace(0, self.lc_sizes-1, num=self.lc_sizes) / (self.lc_sizes-1)
                        m = self.min_size * (self.max_size/self.min_size) ** pw
                        # # m = 2 ** np.linspace(self.min_size, self.max_size, self.lc_sizes)
                        # m = np.array( [int(i) for i in m] )
                        # self.tr_sizes = m
                        # self.print_fn('\nTrain sizes: {}\n'.format(self.tr_sizes))
                        # return None

            # m = np.array( [int(i) for i in m] ) # cast to int

            # # Set min size
            # idx_min = np.argmin( np.abs( m - self.min_size ) )
            # if m[idx_min] > self.min_size:
            #     m = m[idx_min:]  # all values larger than min_size
            #     m = np.concatenate( (np.array([self.min_size]), m) )  # preceed arr with specified min_size
            # else:
            #     m = m[idx_min:]

            # # Set max size
            # idx_max = np.argmin( np.abs( m - self.max_size ) )
            # if m[idx_max] > self.max_size:
            #     m = list(m[:idx_max])    # all values EXcluding the last one
            #     m.append(self.max_size)
            # else:
            #     m = list(m[:idx_max+1])  # all values INcluding the last one
            #     m.append(self.max_size) # TODO: should we append this??
            #     # If the diff btw max_samples and the latest sizes (m[-1] - m[-2]) is "too small",
            #     # then remove max_samples from the possible sizes.
            #     if 0.5*m[-3] > (m[-1] - m[-2]): m = m[:-1] # heuristic to drop the last size

            m = np.array( [int(i) for i in m] ) # cast to int
            self.tr_sizes = m
        # --------------------------------------------

        self.print_fn('\nTrain sizes: {}\n'.format(self.tr_sizes))
        return None


    def trn_learning_curve(self,
            framework: str,

            ml_model_def,
            ml_init_args: dict={},
            ml_fit_args: dict={},
            data_prep_def=None,

            ps_hpo_dir: str=None,

            keras_callbacks_def=None,
            # keras_callbacks_kwargs: dict={},
            keras_clr_args: dict={},

            ## metrics: list=['r2', 'neg_mean_absolute_error'],
            n_jobs: int=4,
            plot=True):
        """
        Args:
            framework : ml framework (keras, lightgbm, or sklearn)
            mltype : type to ml problem (reg or cls)
            ml_model_def : func than create the ml model
            ml_init_args : dict of parameters that initializes the estimator
            ml_fit_args : dict of parameters to the estimator's fit() method
            data_prep_def : func that prepares data for keras model
            keras_clr_args :
            metrics : allow to pass a string of metrics  TODO!
        """
        self.framework = framework

        self.ml_model_def = ml_model_def
        self.ml_init_args = ml_init_args
        self.ml_fit_args = ml_fit_args
        self.data_prep_def = data_prep_def

        if ps_hpo_dir is not None:
            from utils.k_tuner import read_hp_prms
            self.ps_hpo_dir = Path(ps_hpo_dir)
            files = glob( str(self.ps_hpo_dir/'tr_sz*') )
            hp_sizes = {int(f.split(os.sep)[-1].split('tr_sz_')[-1]): Path(f) for f in files}
            self.hp_sizes = {k: hp_sizes[k] for k in sorted(list(hp_sizes.keys()))} # sort dict by key
        else:
            self.hp_sizes = None

        self.keras_callbacks_def = keras_callbacks_def
        # self.keras_callbacks_kwargs = keras_callbacks_kwargs
        self.keras_clr_args = keras_clr_args

        ## self.metrics = metrics
        self.n_jobs = n_jobs

        # Start nested loop of train size and cv folds
        tr_scores_all = []  # list of dicts
        vl_scores_all = []  # list of dicts
        te_scores_all = []  # list of dicts

        # Record runtime per size
        runtime_records = []

        # CV loop
        for split_num in self.tr_dct.keys():
            self.print_fn(f'Split {split_num} out of {list(self.tr_dct.keys())}')

            # Get the indices for this split
            tr_id = self.tr_dct[split_num]
            vl_id = self.vl_dct[split_num]
            te_id = self.te_dct[split_num]

            # Extract Train set T, Validation set V, and Test set E
            # Samples from xtr are sequentially sampled for TRAIN
            # Fixed set of VAL samples for the current CV split
            # Fixed set of TEST samples for the current CV split
            xtr_df, ytr_df, mtr_df = self.get_data_by_id(tr_id)
            xvl_df, yvl_df, mvl_df = self.get_data_by_id(vl_id)
            xte_df, yte_df, mte_df = self.get_data_by_id(te_id)

            # New
            yvl = np.asarray(yvl_df)
            yte = np.asarray(yte_df)

            # Loop over subset sizes (iterate across the dataset sizes and train)
            for i, tr_sz in enumerate(self.tr_sizes):
                # For each size: train model (and save) model; calc tr, vl and te scores
                self.print_fn(f'\tTrain size: {tr_sz} ({i+1}/{len(self.tr_sizes)})')

                # Sequentially get a subset of samples (the input dataset X must be shuffled)
                xtr_sub_df = xtr_df.iloc[:tr_sz, :]
                ytr_sub_df = ytr_df.iloc[:tr_sz]
                mtr_sub_df = mtr_df.iloc[:tr_sz, :]

                # New
                ytr_sub = np.asarray(ytr_sub_df)

                # HP set per tr size
                if self.hp_sizes is not None:
                    keys_vec = list(self.hp_sizes.keys())
                    idx_min = np.argmin( np.abs( keys_vec - tr_sz ) )
                    hp_path = self.hp_sizes[ keys_vec[idx_min] ]
                    hp_path = hp_path/'best_hps.txt'
                    ml_init_args = read_hp_prms(hp_path)
                    ml_init_args.update(self.ml_init_args)

                if self.data_prep_def is not None:
                    xtr_sub = self.data_prep_def(xtr_sub_df)
                    xvl = self.data_prep_def(xvl_df)
                    xte = self.data_prep_def(xte_df)
                else:
                    xtr_sub = np.asarray(xtr_sub_df)
                    xvl = np.asarray(xvl_df)
                    xte = np.asarray(xte_df)

                # Get the estimator
                model = self.ml_model_def(**ml_init_args)

                # Train
                eval_set = (xvl, yvl)
                if self.framework == 'lightgbm':
                    model, trn_outdir, runtime = self.trn_lgbm_model(
                        model=model, xtr_sub=xtr_sub, ytr_sub=ytr_sub,
                        split=split_num, tr_sz=tr_sz, eval_set=eval_set)

                elif self.framework == 'sklearn':
                    model, trn_outdir, runtime = self.trn_sklearn_model(
                        model=model, xtr_sub=xtr_sub, ytr_sub=ytr_sub,
                        split=split_num, tr_sz=tr_sz, eval_set=None)

                elif self.framework == 'keras':
                    model, trn_outdir, runtime = self.trn_keras_model(
                        model=model, xtr_sub=xtr_sub, ytr_sub=ytr_sub,
                        split=split_num, tr_sz=tr_sz, eval_set=eval_set)

                elif self.framework == 'pytorch':
                    raise ValueError(f'Framework {self.framework} is not supported.')

                else:
                    raise ValueError(f'Framework {self.framework} is not supported.')

                if model is None:
                    continue  # if keras fails to train a model (produces nan)

                # Dump args
                # model_args = self.ml_init_args.copy()
                model_args = ml_init_args.copy()
                model_args.update(self.ml_fit_args)
                dump_dict(model_args, trn_outdir/'model_args.txt')

                # Save plot of target distribution
                # plot_hist(ytr_sub, title=f'(Train size={tr_sz})', path=trn_outdir/'hist_tr.png')
                # plot_hist(yvl, title=f'(Val size={len(yvl)})', path=trn_outdir/'hist_vl.png')
                # plot_hist(yte, title=f'(Test size={len(yte)})', path=trn_outdir/'hist_te.png')

                # Calc preds and scores
                # ... training set
                # y_pred, y_true = calc_preds(model, x=xtr_sub, y=ytr_sub, mltype=self.mltype)
                # tr_scores = calc_scores(y_true=y_true, y_pred=y_pred, mltype=self.mltype, metrics=None)
                # dump_preds(y_true, y_pred, meta=mtr_sub_df, outpath=trn_outdir/'preds_tr.csv')
                # ... val set
                # y_pred, y_true = calc_preds(model, x=xvl, y=yvl, mltype=self.mltype)
                # vl_scores = calc_scores(y_true=y_true, y_pred=y_pred, mltype=self.mltype, metrics=None)
                # dump_preds(y_true, y_pred, meta=mvl_df, outpath=trn_outdir/'preds_vl.csv')
                # ... test set
                y_pred, y_true = calc_preds(model, x=xte, y=yte, mltype=self.mltype)
                te_scores = calc_scores(y_true=y_true, y_pred=y_pred, mltype=self.mltype, metrics=None)
                dump_preds(y_true, y_pred, meta=mte_df, outpath=trn_outdir/'preds_te.csv')

                del model

                # Store runtime
                runtime_records.append((split_num, tr_sz, runtime/3600))

                # Add metadata
                # tr_scores['set'] = 'tr'
                # tr_scores['split'] = 'split' + str(split_num)
                # tr_scores['tr_size'] = tr_sz

                # vl_scores['set'] = 'vl'
                # vl_scores['split'] = 'split' + str(split_num)
                # vl_scores['tr_size'] = tr_sz

                te_scores['set'] = 'te'
                te_scores['split'] = 'split' + str(split_num)
                te_scores['tr_size'] = tr_sz

                # Append scores (dicts)
                # tr_scores_all.append(tr_scores)
                # vl_scores_all.append(vl_scores)
                te_scores_all.append(te_scores)

                # Dump intermediate scores
                # scores = pd.concat([scores_to_df([tr_scores]),
                #                     scores_to_df([vl_scores]),
                #                     scores_to_df([te_scores])], axis=0)
                scores = scores_to_df([te_scores])  # (new!)
                scores.to_csv(trn_outdir/'scores.csv', index=False)
                del trn_outdir, scores

        # Scores to df
        # tr_scores_df = scores_to_df(tr_scores_all)
        # vl_scores_df = scores_to_df(vl_scores_all)
        te_scores_df = scores_to_df(te_scores_all)
        # lc_scores = pd.concat([tr_scores_df, vl_scores_df, te_scores_df], axis=0)
        lc_scores = scores_to_df(te_scores_all)  # (new!)

        # Dump final results
        # tr_scores_df.to_csv(self.outdir/'tr_lc_scores.csv', index=False)
        # vl_scores_df.to_csv(self.outdir/'vl_lc_scores.csv', index=False)
        te_scores_df.to_csv(self.outdir/'te_lc_scores.csv', index=False)

        # Runtime df
        runtime_df = pd.DataFrame.from_records(
            runtime_records, columns=['split', 'tr_sz', 'time(hrs)'])
        runtime_df.to_csv(self.outdir/'runtime.csv', index=False)

        return lc_scores


    def get_data_by_id(self, idx):
        """ Returns a tuple of (features (x), target (y), metadata (m))
        for an input array of indices (idx). """
        # x_data = self.X[idx, :]
        # y_data = np.squeeze(self.Y[idx, :])        
        # m_data = self.meta.loc[idx, :]
        # x_data = self.X.loc[idx, :].reset_index(drop=True)
        # y_data = np.squeeze(self.Y.loc[idx, :]).reset_index(drop=True)
        # m_data = self.meta.loc[idx, :].reset_index(drop=True)
        # return x_data, y_data, m_data
        x_data = self.X.iloc[idx, :].reset_index(drop=True)
        y_data = np.squeeze(self.Y.iloc[idx, :]).reset_index(drop=True)
        if self.meta is not None:
            m_data = self.meta.iloc[idx, :].reset_index(drop=True)
        else:
            meta = None
        return x_data, y_data, m_data    


    # def trn_keras_model(self, model, xtr_sub, ytr_sub, split, tr_sz, eval_set=None):
    def trn_keras_model(self, model, xtr_sub, ytr_sub, split, tr_sz, eval_set=None):
        """ Train and save Keras model. """
        trn_outdir = self.create_trn_outdir(split, tr_sz)

        # Fit params
        ml_fit_args = self.ml_fit_args.copy()
        ml_fit_args['validation_data'] = eval_set
        ml_fit_args['callbacks'] = self.keras_callbacks_def(
                outdir=trn_outdir, # **self.keras_callbacks_kwargs,
                **self.keras_clr_args)

        # Train model
        t0 = time()
        history = model.fit(xtr_sub, ytr_sub, **ml_fit_args)
        runtime = time() - t0
        save_krs_history(history, outdir=trn_outdir)
        plot_prfrm_metrics(history, title=f'Train size: {tr_sz}', skp_ep=10,
                           add_lr=True, outdir=trn_outdir)

        # Remove key (we'll dump this dict so we don't need to print all the eval set)
        # ml_fit_args.pop('validation_data', None)
        # ml_fit_args.pop('callbacks', None)

        # Load the best model (https://github.com/keras-team/keras/issues/5916)
        # model = keras.models.load_model(str(trn_outdir/'model_best.h5'), custom_objects={'r2_krs': ml_models.r2_krs})
        model_path = trn_outdir/'model_best.h5'
        if model_path.exists():
            # model = keras.models.load_model( str(model_path) )
            import tensorflow as tf
            model = tf.keras.models.load_model(str(model_path), custom_objects={'r2_krs': r2_krs})
        else:
            model = None
        return model, trn_outdir, runtime


    def trn_lgbm_model(self, model, xtr_sub, ytr_sub, split, tr_sz, eval_set=None):
        """ Train and save LigthGBM model. """
        trn_outdir = self.create_trn_outdir(split, tr_sz)
        
        # Fit params
        ml_fit_args = self.ml_fit_args.copy()
        ml_fit_args['eval_set'] = eval_set  
        
        # Train and save model
        t0 = time()
        model.fit(xtr_sub, ytr_sub, **ml_fit_args)
        runtime = time() - t0

        # Remove key (we'll dump this dict so we don't need to print all the eval set)
        ml_fit_args.pop('eval_set', None)

        if self.save_model:
            joblib.dump(model, filename = trn_outdir/('model.'+self.model_name+'.pkl') )
        return model, trn_outdir, runtime


    def trn_sklearn_model(self, model, xtr_sub, ytr_sub, split, tr_sz, eval_set=None):
        """ Train and save sklearn model. """
        trn_outdir = self.create_trn_outdir(split, tr_sz)
        
        # Fit params
        ml_fit_args = self.ml_fit_args
        # ml_fit_args = self.ml_fit_args.copy()

        # Train and save model
        t0 = time()
        model.fit(xtr_sub, ytr_sub, **ml_fit_args)
        runtime = time() - t0
        if self.save_model:
            joblib.dump(model, filename = trn_outdir / ('model.'+self.model_name+'.pkl') )
        return model, trn_outdir, runtime


    def create_trn_outdir(self, split, tr_sz):
        trn_outdir = self.outdir / ('split'+str(split) + '_sz'+str(tr_sz))
        os.makedirs(trn_outdir, exist_ok=True)
        return trn_outdir
# --------------------------------------------------------------------------------


def scores_to_df(scores_all):
    """ (tricky commands) """
    df = pd.DataFrame(scores_all)
    df_mlt = df.melt(id_vars=['split', 'tr_size', 'set'])
    df_mlt = df_mlt.rename(columns={'variable': 'metric'})
    df_mlt = df_mlt.rename(columns={'value': 'score'})

    # TODO: the pivoting complicates things; consider keep it melted
    # df = df_mlt.pivot_table(index=['metric', 'tr_size', 'set'], columns=['split'], values='score')
    # df = df.reset_index(drop=False)
    # df.columns.name = None
    # return df
    return df_mlt
