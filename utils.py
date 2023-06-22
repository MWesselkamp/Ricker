import numpy as np
from simulations import Simulator, simulate_temperature

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

# Create predictions and observations
def generate_data(timesteps=50, growth_rate = 0.05,
                  sigma = 0.00, phi = 0.00, initial_uncertainty = 0.00,
                  doy_0 = 0, initial_size=1, ensemble_size = 10, environment = "exogeneous",
                  add_trend=False, add_noise=False):

    sims = Simulator(model_type="single-species",
                     environment=environment,
                     growth_rate=growth_rate,
                     ensemble_size=ensemble_size,
                     initial_size=initial_size)
    exogeneous = simulate_temperature(365+timesteps, add_trend = add_trend, add_noise = add_noise)
    exogeneous = exogeneous[365+doy_0:]
    xpreds = sims.simulate(sigma= sigma,phi= phi,initial_uncertainty=initial_uncertainty, exogeneous = exogeneous)['ts_obs']

    obs = Simulator(model_type="multi-species",
                    environment=environment,
                    growth_rate=growth_rate,
                    ensemble_size=1,
                    initial_size=(initial_size, initial_size))
    exogeneous = simulate_temperature(365+timesteps, add_trend = add_trend, add_noise = add_noise)
    exogeneous = exogeneous[365+doy_0:]
    xobs = obs.simulate(sigma= sigma,phi= phi,initial_uncertainty=initial_uncertainty, exogeneous = exogeneous)['ts_obs']

    return xpreds, xobs

generate_data()

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