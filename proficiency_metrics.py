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
