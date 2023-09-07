import numpy as np
import os
from datetime import datetime

def create_experiment_folder(directory_path):
    try:
        # Get the current date and time
        current_datetime = datetime.now()
        # Generate a timestamp in the "yymmdd_hhss" format
        timestamp = current_datetime.strftime("%y%m%d_%H%M")
        # Create a new folder with the timestamp as the name
        new_folder_name = f"version_{timestamp}"
        new_folder_path = os.path.join(directory_path, new_folder_name)
        os.makedirs(new_folder_path)

        print(f"Created experiment folder: {new_folder_path}")

        return new_folder_path

    except Exception as e:
        print(f"An error occurred: {str(e)}")

def create_scenario_folder(directory_path, new_folder_name):

    try:
        new_folder_path = os.path.join(directory_path, new_folder_name)

        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
            print(f"Created scenario folder: {new_folder_path}")
        else:
            print(f"Folder already exists: {new_folder_path}")

        return new_folder_path

    except Exception as e:
        print(f"An error occurred: {str(e)}")


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

def efh_lyapunov(lyapunovs, Delta, delta, fix=False):
    """
    Calculate the forecast horizon with the Lyapunov time (Petchey 2015).
    :param lyapunovs:
    :param Delta: Accuracy/Precision threshold
    :param delta: Accuracy at inital conditions
    :return: EFH
    """
    # np.multiply.outer(1/lyapunovs,np.log(precision/dell0))
    if fix:
        return np.multiply(1 / lyapunovs, np.log(fix))
    else:
        return np.multiply(1/lyapunovs,np.log(Delta/delta))

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

def simulate_temperature(timesteps, add_trend=False, add_noise=False):

    x = np.arange(timesteps)
    freq = timesteps / 365
    y = np.sin(2 * np.pi * freq * (x / timesteps))

    if add_trend:
        y = y + np.linspace(0, 0.1, timesteps)

    if add_noise:
        y = np.random.normal(y, 0.2)
    y = np.round(y, 4)

    return y

def standardize(y):

    ys = (y - np.mean(y))/np.std(y)

    return ys

def sample_ensemble_member(ensemble):

    index = np.random.randint(0, ensemble.shape[0], 1)
    control = ensemble[index, :]
    ensemble = np.delete(ensemble, index, axis=0)

    return control, ensemble, index
