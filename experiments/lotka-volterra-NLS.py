import numpy as np
from scipy.optimize import leastsq, least_squares
import matplotlib.pyplot as plt

from utils import simulate_temperature
from models import Ricker_Predation

def simple_model(Temp, alpha1, beta1, gamma1, bx1, cx1, alpha2, beta2, gamma2, bx2, cx2, sigma):

    n = len(Temp)
    out = np.ones((2, n))

    for i in range(n - 1):
        out[0, i + 1] = out[0, i] * np.exp(alpha1 * (1 - beta1 * out[0, i] - gamma1 * out[1, i] + bx1 * Temp[i] + cx1 * Temp[i]**2)) \
                        + sigma * np.random.normal(loc=0.0, scale=0.1)
        out[1, i + 1] = out[1, i] * np.exp(alpha2 * (1 - beta2 * out[1, i] - gamma2 * out[0, i] + bx2 * Temp[i] + cx2 * Temp[i]**2)) \
                         + sigma * np.random.normal(loc=0.0, scale=0.1)

    return out

def simple_model_2(Temp, alpha, beta, bx, cx):

    n = len(Temp)
    out = np.ones_like(Temp, dtype=np.double)

    for i in range(n - 1):
        # Introduce randomness
        #epsilon = sigma * np.random.normal(loc=0.0, scale=1.0)

        out[i + 1] = out[i] * np.exp(alpha * (1 - beta * out[i] + bx * Temp[i] + cx * Temp[i] ** 2)) #+ epsilon

    return out

# Generate some simulated data using simple_model
np.random.seed(42)  # Set seed for reproducibility
x = np.round(simulate_temperature(500, freq_s=1000, add_trend=True, add_noise=False), 4)
x_train, x_test = x[:250], x[250:]
# simulate observations from Ricker_predation model
y = simple_model(x, 0.03, 1, -0.02, 0.1, -0.05, 0.06, 1, 0.03, 0.02, 0.1, sigma=0.05)
y_train, y_test = y[:, :250], y[:, 250:]
plt.plot(y.T, color='grey')
plt.plot(y_train.T, color='red')
plt.plot(y_test.T, color='blue')


# Define the objective function to minimize (residuals)
def residuals(params, x, simulated_data):
    alpha, beta, bx, cx = params
    model_data = simple_model_2(x, alpha, beta, bx, cx)
    return simulated_data - model_data


# Initial guess for parameters
initial_params = [0.1, 0.2, 0.3, 0.4]

# Fit the simple_model_2 to the simulated data using least squares
result = leastsq(residuals, initial_params, args=(x_train, y_train[0,:]))

# Extract the fitted parameters
fitted_params = result[0]

# Print the fitted parameters
print("Fitted Parameters:")
print("alpha:", fitted_params[0])
print("beta:", fitted_params[1])
print("bx:", fitted_params[2])
print("cx:", fitted_params[3])

# Plot the fitted model
fitted_model = simple_model_2(x, *fitted_params)

plt.plot(y[0,:], color = 'grey')
plt.plot(fitted_model, color = 'red')
plt.show()

result.fu