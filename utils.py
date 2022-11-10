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


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

def add_identity(axes, *line_args, **line_kwargs):
    # https://stackoverflow.com/questions/22104256/does-matplotlib-have-a-function-for-drawing-diagonal-lines-in-axis-coordinates
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes