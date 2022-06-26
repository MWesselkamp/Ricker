import numpy as np

def lyapunov(timeseries_derivative):
    return np.log(abs(timeseries_derivative))

def lyapunov_efh(lyapunovs, Delta, delta):
    """
    Calculate the forecast horizon with the Lyapunov time.
    :param lyapunovs:
    :param Delta: Accuracy/Precision threshold
    :param delta: Accuracy at inital conditions
    :return: EFH
    """
    # np.multiply.outer(1/lyapunovs,np.log(precision/dell0))
    return np.multiply(1/lyapunovs,np.log(Delta/delta))

