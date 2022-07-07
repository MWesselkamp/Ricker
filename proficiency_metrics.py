import numpy as np
import scipy.special as special

def absolute_difference(ts_reference, ts_ensemble, mean = False):
    absolute_differences = abs(np.subtract(ts_reference, ts_ensemble))
    if mean:
        return absolute_differences, np.mean(absolute_differences, axis=0)
    else:
        return absolute_differences

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
