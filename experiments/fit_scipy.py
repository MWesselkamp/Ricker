import torch
import numpy as np
from scipy.optimize import minimize
from utils import simulate_temperature

from models import Ricker, Ricker_Predation

# Create example data (replace with your actual data)
# simulate temperature data
Temp = np.round(simulate_temperature(365*2, add_trend=True, add_noise=True), 4)
# simulate observations from Ricker_predation model
ricker_predation = Ricker_Predation(params=[0.18, 1, -0.1, 0.41, 0.25, 0.16, 1, 0.3, 0.32, 0.32], noise=None)
#N0 = torch.tensor([1.0, 1.0], dtype=torch.double)
observations = ricker_predation.forward(Temp)

Temp = torch.tensor(Temp, dtype=torch.double)
N0 = torch.tensor([1.0], dtype=torch.double)

# Objective function to minimize (difference between model predictions and actual data)
def mean_squared_error(params, N0, Temp, observations):
    ricker_model = Ricker(params)
    predictions = ricker_model.forward(N0, Temp)
    return torch.sum((predictions - observations[0,:])**2).item()

def negative_log_likelihood(params, N0, Temp, observations):

    alpha, beta, bx, cx, sigma = params
    sigma = torch.tensor(sigma, requires_grad=True, dtype=torch.double)

    ricker_model = Ricker([alpha, beta, bx, cx])
    predictions = ricker_model.forward(N0, Temp, sigma)

    # Calculate the log-likelihood of observing the data given the model
    log_likelihood = -0.5 * torch.sum(torch.log(2 * np.pi * sigma**2) + (observations - predictions)**2 / sigma**2)

    return -log_likelihood.item()  # Return negative log-likelihood for minimization


# Initial parameter values
initial_params = [1.0, 1.0, 1.0, 1.0, 1.0]
initial_params_stoch = [1.0, 1.0, 1.0, 1.0, 1.0]

# Minimize the objective function
result = minimize(mean_squared_error, x0= initial_params, args=(N0, Temp, observations), method='Nelder-Mead')
result = minimize(negative_log_likelihood,x0= initial_params_stoch, args=(N0, Temp, observations), method='Nelder-Mead')

# Extract the optimized parameters
optimized_params = result.x
print("Optimized parameters:")
print(optimized_params)

# Create the model with the optimized parameters
ricker_model = Ricker(optimized_params)

# Make predictions with the model
predictions = ricker_model.forward(N0, Temp)
print("Predictions:")
print(predictions)

# plot predictions
import matplotlib.pyplot as plt
plt.plot(predictions.detach().numpy())
plt.show()
#