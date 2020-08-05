from pathlib import Path
import sklearn
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def calc_preds(model, x, y, mltype):
    """ Calc predictions. """
    if mltype == 'cls': 
        def get_pred_fn(model):
            if hasattr(model, 'predict_proba'):
                return model.predict_proba
            if hasattr(model, 'predict'):
                return model.predict

        pred_fn = get_pred_fn(model)
        if (y.ndim > 1) and (y.shape[1] > 1):
            y_pred = pred_fn(x)
            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.argmax(ydata, axis=1)
        else:
            y_pred = pred_fn(x)
            y_true = y
            
    elif mltype == 'reg':
        y_pred = np.squeeze(model.predict(x))
        y_true = np.squeeze(y)

    return y_pred, y_true


def dump_preds(y_true, y_pred, meta=None, outpath='./preds.csv'):
    """ Dump prediction and true values, with optional with metadata. """
    y_true = pd.Series(y_true, name='y_true')
    y_pred = pd.Series(y_pred, name='y_pred')
    if meta is not None:
        # preds = meta.copy()
        # preds.insert(loc=3, column='y_true', value=y_true.values)
        # preds.insert(loc=4, column='y_pred', value=y_pred.values)
        preds = pd.concat([meta, y_true, y_pred], axis=1)
    else:
        preds = pd.concat([y_true, y_pred], axis=1)
    preds.to_csv(Path(outpath), index=False)


def calc_scores(y_true, y_pred, mltype, metrics=None):
    """ Create dict of scores.
    Args:
        metrics : TODO allow to pass a string of metrics
    """
    scores = {}

    if mltype == 'cls':    
        # Metric that accept probabilities
        scores['brier'] = sklearn.metrics.brier_score_loss(y_true, y_pred, sample_weight=None, pos_label=1)
        scores['auc_roc'] = sklearn.metrics.roc_auc_score(y_true, y_pred)

        # Metric that don't accept probabilities
        y_pred_ = [1 if v>0.5 else 0 for v in y_pred]
        scores['mcc'] = sklearn.metrics.matthews_corrcoef(y_true, y_pred_, sample_weight=None)
        scores['f1_score'] = sklearn.metrics.f1_score(y_true, y_pred_, average='micro')
        scores['acc_blnc'] = sklearn.metrics.balanced_accuracy_score(y_true, y_pred_)

    elif mltype == 'reg':
        scores['r2'] = sklearn.metrics.r2_score(y_true=y_true, y_pred=y_pred)
        scores['mean_absolute_error']   = sklearn.metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)
        scores['median_absolute_error'] = sklearn.metrics.median_absolute_error(y_true=y_true, y_pred=y_pred)
        scores['mse']  = sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)
        scores['rmse'] = scores['mse'] ** 0.5
        # scores['auroc_reg'] = reg_auroc(y_true=y_true, y_pred=y_pred)
        scores['spearmanr'] = spearmanr(y_true, y_pred)[0]
        scores['pearsonr'] = pearsonr(y_true, y_pred)[0]
        
    scores['y_avg_true'] = np.mean(y_true)
    scores['y_avg_pred'] = np.mean(y_pred)
    
    # # https://scikit-learn.org/stable/modules/model_evaluation.html
    # for metric_name, metric in metrics.items():
    #     if isinstance(metric, str):
    #         scorer = sklearn.metrics.get_scorer(metric_name) # get a scorer from string
    #         scores[metric_name] = scorer(ydata, pred)
    #     else:
    #         scores[metric_name] = scorer(ydata, pred)    
    return scores


# def scores_to_df(scores_all):
#     """ Dict to df. """
#     df = pd.DataFrame(scores_all)
#     df = df.melt(id_vars=['run'])
#     df = df.rename(columns={'variable': 'metric'})
#     df = df.pivot_table(index=['run'], columns=['metric'], values='value')
#     df = df.reset_index(drop=False)
#     df.columns.name = None
#     return df



