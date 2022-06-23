import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def lyapunov_efh(lyapunovs, precision, dell0):
    return np.multiply.outer(1/lyapunovs,np.log(precision/dell0))


