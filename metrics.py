import numpy as np
import scipy.special as special
import math
from scipy.stats import pearsonr

def absolute_differences(reference, ensemble, mean = False):
    absolute_differences = abs(np.subtract(reference, ensemble))
    return absolute_differences

def rolling_bias(reference, ensemble):

    b = []
    for i in range(reference.shape[1]):
        b.append(np.mean(np.subtract(reference[0,:i], ensemble[:,:i]), axis=1))
    b = np.transpose(np.array(b))

    return b

def rolling_mse(reference, ensemble):

    """
    Calculates forecast rmse for a time series of predictions by stepwise adding the next time step.
    Change this to a moving window? Or pointwise?
    :param preds: predicted time series
    :return: time series of rmse
    """
    mses = []
    for i in range(reference.shape[1]):
        mse = np.mean(np.subtract(reference[0,:i], ensemble[:, :i])**2, axis=1)
        mses.append(mse)
    return np.transpose(np.array(mses))

def rolling_rmse(reference, ensemble, standardized = False):

    """
    Calculates forecast rmse for a time series of predictions by stepwise adding the next time step.
    Change this to a moving window? Or pointwise?
    :param preds: predicted time series
    :return: time series of rmse
    """
    rmses = []
    for i in range(reference.shape[1]):
        rmse = np.sqrt(np.mean(np.subtract(reference[0,:i], ensemble[:, :i])**2, axis=1))
        rmses.append(rmse)
    return np.transpose(np.array(rmses))

def rolling_corrs(reference, ensemble, window = 3, abs = False):
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
    for j in range(reference.shape[1]-window):
        ecorrs = []
        for i in range(ensemble.shape[0]):
            ecorrs.append(pearsonr(reference[0,j:j+window], ensemble[i,j:j+window])[0])
        corrs.append(ecorrs)
    corrs = np.transpose(np.array(corrs))
    if abs:
        return abs(corrs)
    else:
        return corrs


def rolling_rsquared(y, x):

    r_sq = []
    for i in range(1,y.shape[1]):
        y_mean = np.mean(y[0,:i])
        ss_res = np.sum(np.subtract(y[0,:i], x[:,:i])**2, axis=1)
        ss_tot = np.sum(np.subtract(y[0,:i], y_mean)**2)
        rs = 1 - ss_res/ss_tot
        r_sq.append(rs)
    r_sq = np.transpose(np.array(r_sq))
    return r_sq

def rmse(reference, ensemble, standardized = False):
    if standardized:
        return math.sqrt(np.square(np.subtract(reference, ensemble)).mean()) / (np.max(reference) - np.min(reference))
    else:
        return math.sqrt(np.square(np.subtract(reference,ensemble)).mean())

def mse(reference, ensemble):
    mse = np.mean(np.subtract(reference, ensemble)**2, axis=0)
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

