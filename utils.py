import matplotlib.pyplot as plt
import numpy as np

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

def simulate_T(len, add_trend = False, add_noise = False, show=True):

    x = np.arange(len)
    freq = len/52
    y = np.sin(2 * np.pi * freq * (x / len))

    if add_trend:
        t = np.linspace(0,0.5,len)
        y = y+t

    if add_noise:
        y = np.random.normal(y, 0.1)

    y = np.round(y, 4)

    if show:
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.stem(x, y, 'r')
        plt.plot(x, y)
        ax.set_xlabel('Time step t', size=14)
        ax.set_ylabel('Simulated temperature', size=14)
        fig.show()

    return y


