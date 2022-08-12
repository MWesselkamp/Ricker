import matplotlib.pyplot as plt
import numpy as np

def historic_mean(x_test, x_train, length = 'test'):

    if length=='test':
        length = x_test.shape[0]
    historic_mean = np.full((length), np.mean(x_train), dtype=np.float)
    historic_var = np.full((length), np.std(x_train), dtype=np.float)
    return historic_mean, historic_var

def create_quantiles(n, max):
    u = 0.5 + np.linspace(0, max, n+1)
    l = 0.5 - np.linspace(0, max, n+1)
    r = np.array((l, u)).T
    return r[1:,]

def min_Delta(initial_uncertainty_estimate):
    min_D = initial_uncertainty_estimate/np.exp(-1)
    return min_D

def fixed_Tp_Delta(lyapunovs, Tp, delta):
    Tp_Delta = np.array([delta*np.exp(lya*Tp) for lya in lyapunovs])
    return Tp_Delta

def fixed_Tp_delta(lyapunovs, Tp, Delta):
    Tp_delta = np.array([Delta/np.exp(lya*Tp) for lya in lyapunovs])
    return Tp_delta

def simulate_T(len, add_trend = False, sample_rate = 100, freq = 10):

    x = np.arange(len)
    y = np.sin(2 * np.pi * freq * (x / sample_rate))

    if add_trend:
        t = np.linspace(0,1.5,len)
        y = y+t

    fig = plt.figure()
    plt.stem(x, y, 'r')
    plt.plot(x, y)
    fig.show()

    return y