# ------------------------------------------------------------------------
#   New
# --------------------

# library(reshape2)
if(!require(dplyr)){
  install.packages("dplyr")
  library(dplyr)
}

if(!require(tibble)){
  install.packages("tibble")
  library(tibble)
}

# if(!require(MASS)){
#   install.packages("MASS")
#   library(MASS)
# }

if(!require(minpack.lm)){
  install.packages("minpack.lm")
  library(minpack.lm)
}


summary_coefs <- function(model) {
  coefs <- data.frame(unclass(summary(model))$parameters,
                      check.names=FALSE, stringsAsFactors=FALSE)
  colnames(coefs) <- c('est', 'se', 't_val', 'p_val')
  coefs <- tibble::rownames_to_column(coefs, var='coef')
  rownames(coefs) <- NULL
  return(coefs)
}


# fit_nlsLM_power_law <- function(dfit, startParams=list(a=1.2, b=-0.5, c=0.06)) {
# fit_nlsLM_power_law <- function(x, y, w, startParams=list(a=1.2, b=-0.3, c=0.03)) {
fit_nlsLM_power_law <- function(x, y, w, a=1.2, b=-0.3, c=0.03) {
  # www.r-bloggers.com/a-better-nls/
  # stackoverflow.com/questions/18364402
  
  # The deriv3() is required to compute se.
  # https://r.789695.n4.nabble.com/Using-deriv3-in-a-separated-nonlinear-regression-model-td3551423.html
  # https://docs.tibco.com/pub/enterprise-runtime-for-R/5.0.0/doc/html/Language_Reference/stats/deriv.html
  gradF <- deriv3(~a*(x^b) + c, c('a','b','c'), function(a,b,c,x) NULL);
  dfit <- data.frame(x=x, y=y, w=w)
                  
  # startParams; New!
  # startParams = list(a=1.2, b=-0.3, c=0.03)
  startParams = list(a=b, b=b, c=c)
  
  # Fit model with nls
  model <- minpack.lm::nlsLM(
    formula = y~gradF(a, b, c, x),
    # formula = y ~ a*(x^b) + c,
    start = startParams,  # named list or named numeric vector of starting estimates
    weights = w,          # optional numeric vector of (fixed) weights
    control = nls.control(warnOnly=TRUE,
                          maxiter=100000,
                          tol=1e-4,
                          minFactor=1e-7),
    # algorithm = 'port',
    # lower = lwr_bnd_prms, # lower bounds of parameters
    # upper = upr_bnd_prms, # upper bounds of parameters
    data = dfit)
  
  coef_est <- summary_coefs(model)
  return(coef_est)
}


# add_weight_col <- function(data, type='binomial') {
#   if (type=='binomial') {
#     data$w <- data$x / (data$y * ( 1 - data$y )) # bionomial 
#   } else if (type=='size') {
#     data$w <- data$x # by sample size
#     data$w <- data$w/max(data$w) # TODO: should we normalize weights??
#   } else if (type=='none') {  
#     data$w <- rep(x=1.0, times=length(data$x)) # constant weight
#   }
#   return (data)
# }
                  
# ------------------------------------------------------------------------


# ------------------------------------------------------------------------
#   Old
# --------------------
                  
# if(!require(broom)){
#   install.packages("broom")
#   library(broom)
# }

# if(!require(dplyr)){
#   install.packages("dplyr")
#   library(dplyr)
# }

# # self-defined self-starting fct
# # for more details plz refer to 
# # Predicting accuracy on large datasets from smaller pilot data
# weightfn_selfstart = function (d) { d$training.examples }

# plaw_selfStart = selfStart(~ a + b*(x**(c-0.5)),
#                            function(mCall, data, LHS) {
#                              xy = sortedXyData(mCall[["x"]], LHS, data);
#                              d = xy %>% mutate(training.examples=x, error=y);
#                              lmFit = lm(error ~ I(1/sqrt(training.examples)), data=d, weights=weightfn_selfstart(d));
#                              coefs = coef(lmFit);
#                              a = coefs[1]; # intercept
#                              b = coefs[2]; # factor
#                              value = c(a, b, 0);
#                              names(value) = mCall[c("a", "b", "c")];
#                              return(value);
#                            },
#                            c("a","b","c"));

# bionomial <- function (d) {
#     d$training.examples / (d$error*(1-d$error))
# }

# get_model <- function (data) {
#   # eplaw models: error = a + b*training.examples^c
#   eplaw_models <- data %>%
#     do(model = nls(error ~ plaw_selfStart(training.examples,a,b,c),
#                    data=.,
#                    control=nls.control(warnOnly=TRUE,
#                                        maxiter=100000,
#                                        tol=1e-4,
#                                        minFactor=1e-7),
#                    weights=bionomial(.)));
#   return(eplaw_models$model[[1]])
# }

# model_param <- function (x, y) {
#   data_train <- do.call(rbind, Map(data.frame, 
#                                    training.examples=x, 
#                                    error=y))
#   model <- get_model(data_train)
#   return (coef(model))
# }
                  
# ------------------------------------------------------------------------