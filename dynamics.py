import numpy as np

def lyapunovs(timeseries_derivative, stepwise = False):
    """
    Calcuclate Lyapunov exponent of time series at iteration limit or stepwise.
    :param timeseries_derivative: ndarray. Shape: (ensemble_size, iterations)
    :param stepwise: stepwise or at limit.
    :return: either vector or matrix of lyapunov exponents.
    """
    if stepwise:
        ts_logabs = np.log(abs(timeseries_derivative))
        lyapunovs = np.array([np.mean(ts_logabs[:,1:i], axis=1) for i in range(ts_logabs.shape[1])])
    else:
        lyapunovs = np.mean(np.log(abs(timeseries_derivative)), axis=1)
    return lyapunovs

def efh_lyapunov(lyapunovs, Delta, delta):
    """
    Calculate the forecast horizon with the Lyapunov time (Petchey 2015).
    :param lyapunovs:
    :param Delta: Accuracy/Precision threshold
    :param delta: Accuracy at inital conditions
    :return: EFH
    """
    # np.multiply.outer(1/lyapunovs,np.log(precision/dell0))
    return np.multiply(1/lyapunovs,np.log(Delta/delta))

