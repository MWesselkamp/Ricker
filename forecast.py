
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('module://backend_interagg')

#=====================================#
# Forecast with the fit model to test #
#=====================================#

# load test data.
# load the model and the model fit (posterior samples).

# Get parameter estimates - here for r_mean, r_sd.

r_posterior = posterior_samples['r'] # Extract posterior distributions for prediction

its = real_test.shape[0]

pops = real_test.shape[1]
samples = len(r_posterior)
forecasts = np.zeros((its, samples*pops))

k = 0
for i in range(pops):
    for j in range(samples):
        forecasts[:,k] = mod.model_iterate(real_test[0,i], r_posterior[j], its-1)
        k += 1

if __name__ == '__main__':
    plt.plot(forecasts)