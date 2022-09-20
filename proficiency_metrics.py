import numpy as np
import scipy.special as special
import math
from scipy.stats import pearsonr

def absolute_difference(ts_reference, ts_ensemble, mean = False):
    absolute_differences = abs(np.subtract(ts_reference, ts_ensemble))
    if mean:
        return absolute_differences, np.mean(absolute_differences, axis=0)
    else:
        return absolute_differences

def mean_squared_error_rolling(ts_reference, ts_ensemble):

    """
    Calculates forecast rmse for a time series of predictions by stepwise adding the next time step.
    Change this to a moving window? Or pointwise?
    :param preds: predicted time series
    :return: time series of rmse
    """
    mses = []
    for i in range(ts_reference.shape[0]):
        mse = np.mean(np.subtract(ts_reference[:i + 1], ts_ensemble[:, :i + 1])**2, axis=1)
        mses.append(mse)
    return np.transpose(np.array(mses))

def rolling_corrs(ts_reference, ts_ensemble, window = 3):
    """
    Rolling correlations between true and predicted dynamics in a moving window.
    Change to cross-correlation?
    :param obs: np.vector. true dynamics
    :param preds: np.array. ensemble predictions.
    :param test_index: int. Where to start calculation
    :param window: int. Size of moving window.
    :return: array with correlations. shape:
    """
    corrs = []
    for j in range(ts_reference.shape[0]-window):
        ecorrs = []
        for i in range(ts_ensemble.shape[0]):
            ecorrs.append(pearsonr(ts_reference[j:j+window], ts_ensemble[i,j:j+window])[0])
        corrs.append(ecorrs)
    corrs = np.transpose(np.array(corrs))
    return corrs


def rmse(y, y_pred):
    return math.sqrt(np.square(np.subtract(y,y_pred)).mean())

def mean_squared_error(ts_reference, ts_ensemble):
    mse = np.mean(np.subtract(ts_reference, ts_ensemble)**2, axis=0)
    return mse

def t_statistic(x_sample, H0):
    """
    Student's t-test. Two-sided.
    """
    df = x_sample.shape[0]-1
    v = np.var(x_sample, axis=0, ddof=1)
    denom = np.sqrt(v/df)
    t = np.divide((x_sample.mean(axis=0)-H0),denom)
    pval = special.stdtr(df, -np.abs(t))*2
    return t, pval

def squared_error_SNR(obs, pred):
    """
    The squared error based SNR, an estimator of the true expected SNR. (Czanner et al. 2015, PNAS)
    The signal is the reduction in expected prediction error by using the model that generated pred.
    Independent of sample size!
    """

    EPE_mean = np.dot(np.transpose(obs - np.mean(obs)), (obs - np.mean(obs))) # returns a scalar
    EPE_model = np.dot(np.transpose(obs - pred), (obs - pred)) # also returns a scalar
    signal = (EPE_mean - EPE_model)
    noise = EPE_model

    return signal/noise


def var_based_SNR(obs, pred, inital_uncertainty):
    """
    The squared error based SNR, an estimator of the true expected SNR at perfect knowledge of parameters.
    The signal is the reduction in expected prediction error by using the model that generated pred.
    Dependent on sample size (decreases with sample size)

    # This may be more suited for the perfect model scenario
    # but I am not going to use it until I am sure of what the individual parts are
    """
    signal = np.dot(np.transpose(pred - np.mean(obs)), (pred - np.mean(obs)))
    noise = len(obs) * inital_uncertainty ** 2

    return signal / noise

def raw_SNR(pred):
    # tSNR raw SNR or timeseries SNR: mean(timeseries) / var(timeseries)
    # tsnr increases with sample size (see sd).
    mu = np.mean(pred, axis=1)
    var = np.mean(pred**2, axis=1) - np.mean(pred, axis=1)**2 # np.std(pred, axis=1)#1/pred.shape[0]*np.sum(np.subtract(pred, mu)**2, axis=0)
    return mu**2/var**2

def CNR(x_c, x_b):
    """
    CNR - contrast to noise ratio: mean(condition-baseline) / std(baseline)
    This is basically the same as the square-error-based SNR?
    Transfered, we have the model as the baseline and the mean as condition.
    tsnr increases with sample size (see sd).
    """
    mu = np.mean(x_c) - np.mean(x_b)
    sd = 1/x.shape[0]*np.sum(np.subtract(x_b, mu)**2, axis=0)
    return mu/sd
