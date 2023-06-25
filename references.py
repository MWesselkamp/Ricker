import numpy as np
from simulations import Simulator, simulate_temperature

def rolling_climatology(observations, predictions = None):

    hmean = np.zeros(observations.shape)
    hvar = np.zeros(observations.shape)
    hmean[:,0] = observations[:,0]

    for i in range(1, observations.shape[1]):
        hmean[:,i] = np.mean(observations[0,:i], axis=0)
        hvar[:,i] = np.std(observations[0,:i], axis=0)

    return hmean

def climatology(observations, predictions = None):

    hmean = np.mean(observations) #, axis=0)
    hvar = np.std(observations)#, axis=0)

    if not predictions is None:
        sh = predictions.shape
        return np.full((1,sh[1]), hmean), np.full((1,sh[1]), hvar)
    else:
        sh = observations.shape
        return np.full(sh, hmean)

def diurnal_climatology(scenario = "perfect_model", growth_rate=0.1, days=365, years=10,
                        sigma = 0.00, phi = 0.00, initial_uncertainty = 0.00,
                        add_trend=False, add_noise = False):
    if scenario =='perfect_model':
        obs = Simulator(model_type="single-species",
                            environment='exogeneous',
                            growth_rate=growth_rate,
                            ensemble_size=1,
                            initial_size=1)
    elif scenario == 'imperfect_model':
        obs = Simulator(model_type="multi-species",
                        environment='exogeneous',
                        growth_rate=growth_rate,
                        ensemble_size=1,
                        initial_size=(1, 1))
    exogeneous = simulate_temperature(days*(years), add_trend=add_trend, add_noise=add_noise)
    xobs = obs.simulate(sigma=sigma, phi=phi, initial_uncertainty=initial_uncertainty, exogeneous=exogeneous)[
            'ts_obs']
    if scenario =='perfect_model':
        xobs = xobs.squeeze()
    elif scenario == 'imperfect_model':
        xobs = xobs[:,:,0].squeeze()

    unique_values = np.arange(days)
    year = np.tile(unique_values, reps=years)
    aggregated_data = np.zeros((days, years))

    for i, value in enumerate(unique_values):
        mask = (year == value)
        aggregated_data[i,:] = xobs[mask] # Aggregate observations by day

    return aggregated_data


def persistence(observations, predictions = None):

    if not predictions is None:
        sh = predictions.shape
    else:
        sh = observations.shape

    x_pred = observations[:,-1]
    return np.full(sh, x_pred)


