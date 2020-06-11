# import rpy2.robjects as robjects
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# ------------------------------------------------------------------------
#   New
# --------------------
def biased_powerlaw(x, alpha, beta, gamma):
    return alpha * x**(beta) + gamma


def fit_model(x: ro.IntVector, y: ro.FloatVector, w: ro.FloatVector):
    """ ... """
    x = ro.IntVector(list(x))
    y = ro.FloatVector(list(y))
    w = ro.FloatVector(list(w))
    
    # script = '\'../fit.R\''
    script = '\'nls_lm.R\''
    ro.r('''source({})'''.format(script))
    fit_nlsLM_power_law = ro.globalenv['fit_nlsLM_power_law']
    coef_est_r = fit_nlsLM_power_law(x, y, w)
    
    # coef_est_py = pandas2ri.ri2py_dataframe(coef_est_r)
    with localconverter(ro.default_converter + pandas2ri.converter):
        coef_est_py = ro.conversion.rpy2py(coef_est_r)
    
    return coef_est_py
# ------------------------------------------------------------------------


# ------------------------------------------------------------------------
#   Old
# --------------------

# def biased_powerlaw(x, alpha, beta, gamma):
#     return alpha * x**(beta) + gamma

# def fit_params(x, y):  
#     x = robjects.IntVector(list(x))
#     y = robjects.FloatVector(list(y))
#     # script = '\'../fit.R\''
#     script = '\'nls_lm.R\''
#     robjects.r('''source({})'''.format(script))
#     get_params = robjects.globalenv['model_param']
#     a, b, c = get_params(x, y)
    
#     prms_dct = {}
#     prms_dct['alpha'] = b
#     prms_dct['beta'] = c-0.5
#     prms_dct['gamma'] = a    
#     return prms_dct

# ------------------------------------------------------------------------