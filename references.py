import numpy as np

def historic_mean(observations):
    """
    Computed as rolling historic mean!
    """
    historic_mean = np.zeros(observations.shape)
    historic_var = np.zeros(observations.shape)
    historic_mean[:,0] = observations[:,0]

    for i in range(1, observations.shape[1]):
        historic_mean[:,i] = np.mean(observations[0,:i], axis=0)
        historic_var[:,i] = np.std(observations[0,:i], axis=0)

    return historic_mean


