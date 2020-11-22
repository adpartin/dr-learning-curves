import rpy2.robjects as robjects
from pathlib import Path
fpath = Path(__file__).parent


def biased_powerlaw(x, alpha, beta, gamma):
    return alpha * x**(beta) + gamma


def fit_params(x, y) -> tuple:
    x = robjects.IntVector(list(x))
    y = robjects.FloatVector(list(y))

    # script = '\'fit.R\''
    script = "\"" + str(fpath/'fit.R') + "\""
    robjects.r('''source({})'''.format(script))
    get_params = robjects.globalenv['model_param']
    a, b, c = get_params(x, y)
    
    prms_dct = {}
    prms_dct['alpha'] = b
    prms_dct['beta'] = c - 0.5
    prms_dct['gamma'] = a
    return prms_dct
    