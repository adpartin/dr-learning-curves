import rpy2.robjects as robjects


# def biased_powerlaw(x, a, b, c):
#     return a + b*(x**(c))


# def params(x: robjects.IntVector, y: robjects.FloatVector) -> tuple:
#     """ This function fit weighted power-law model. """
#     script = '\'fit.R\''
#     robjects.r('''source({})'''.format(script))
#     get_params = robjects.globalenv['model_param']
#     a, b, c = get_params(x, y)
#     return (a, b, c-0.5)


def weighted_power_law(x, alpha, beta, gamma):
    return gamma + alpha*(x**(beta))


# def params(x: robjects.IntVector, y: robjects.FloatVector) -> tuple:
def fit_weighted_pwr_law(x: robjects.IntVector, y: robjects.FloatVector) -> tuple:
    """ This function fit weighted power-law model. """
    # script = '\'fit.R\''
    script = '\'../fit.R\''
    robjects.r('''source({})'''.format(script))
    get_params = robjects.globalenv['model_param']
    try:
        gamma, alpha, beta = get_params(x, y)
        prms_dct = {}
        
        # prms_dct['alpha'], prms_dct['beta'], prms_dct['gamma'] = alpha, beta, gamma-0.5
        prms_dct['alpha'] = alpha
        prms_dct['beta'] = beta
        prms_dct['gamma'] = gamma-0.5
        return prms_dct    
    except:
        print('Could not fit power-law.')

