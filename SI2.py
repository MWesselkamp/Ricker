import calibration as ft
import sympy as sp
import numpy as np
import os
import matplotlib.pyplot as plt

from visualisations import baseplot
from utils import create_scenario_folder

#============================#
# Explore simulated dynamics #
#============================#

process = 'deterministic'
scenario = 'chaotic'
experiment_path = 'results/version_230908_1004'
scenario_path = create_scenario_folder(experiment_path, f'{scenario}_{process}_SI2')

observation_params, initial_params, true_noise, initial_noise = ft.set_parameters(process=process, scenario=scenario,
                                                                                      dir = scenario_path)
#observation_params = [1.18, 1, -0.03, 0.41, 0.25, 1.16, 1, -0.13, 0.62, 0.32]
#dyn_observed, temperature = ft.create_observations(years=1,
#                                    observation_params=observation_params,
#                                    true_noise=0.0, full_dynamics=True)
#baseplot(dyn_observed.transpose()[:, 0], dyn_observed.transpose()[:, 1])

y_train, y_test, sigma_train, sigma_test, x_train, x_test, climatology = ft.create_observations(years=30,
                                                                                                 observation_params=observation_params,
                                                                                                 true_noise=true_noise)


fitted_values = ft.model_fit(False, os.path.join(experiment_path, f'{process}_{scenario}'),
                             y_train, x_train, sigma_train, initial_params, initial_noise,
                             samples=1, epochs=15, loss_fun='mse', step_length=10)
parameter_samples = ft.get_parameter_samples(fitted_values, uncertainty=0.02)
yinit, forecast, modelfits = ft.forecast_fitted(y_test, x_test, parameter_samples, initial_params, initial_noise,
                                                initial_uncertainty=0.01)
baseplot(y_test.detach().numpy(), forecast.transpose())

dyn_observed, temperature = ft.create_observations(years=30,
                                    observation_params=observation_params,
                                    true_noise=true_noise, full_dynamics=True)

baseplot(forecast.transpose(),dyn_observed.transpose()[:365*2, 0], dyn_observed.transpose()[:365*2, 1],
         xlab='Generation', ylab='Relative size', x2lab='Observed', x3lab='Neglected',
         dir = scenario_path, name='full_system_dynamics')

#=============#
# Get Lyapunov #
#=============#
def two_species_lyapunov(dyn_obs, temp, par_values, start, timesteps):

    jacobian_evaluated = np.ones((2,2))

    for i in range(timesteps-start):
        N1 ,N2 = sp.symbols('N1, N2')
        params = sp.symbols('g1 c1 co1 t1 ts1 g2 c2 co2 t2 ts2 Temp')
        equation1 = N1 * sp.exp(params[0]*(1 - params[1]*N1 - params[2]*N2+ params[3] * params[10] + params[4] * params[10] ** 2))
        equation2 = N2 * sp.exp(params[5]*(1 - params[6]*N2 - params[7]*N1 + params[8] * params[10] + params[9] * params[10] ** 2))

        variables = [N1, N2]
        equations = [equation1, equation2]
        jacobian_matrix = sp.Matrix([[sp.diff(eq, var) for var in variables] for eq in equations])
        jacobian_matrix

        jacobian_func = sp.lambdify((N1, N2, params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10]), expr=jacobian_matrix)
        state_values = (dyn_obs[i,0], dyn_obs[i,1])
        param_values = par_values + (temp[i],)

        # Evaluate the Jacobian matrix at the given parameter values and state
        jacobian_evaluated *= jacobian_func(*state_values, *param_values)

    dominant_eigenvalue = max(abs(np.linalg.eigvals(jacobian_evaluated)))
    lyapunov = np.log(dominant_eigenvalue)/timesteps

    return(lyapunov)

l1 = []
l2 = []
timesteps1 = 130
timesteps2 = 110
for i in range(timesteps1):
    start = i
    dyn_obs = dyn_observed.transpose()[start:timesteps1]
    temp = temperature[start:timesteps1]
    l1.append(two_species_lyapunov(dyn_obs, temp, tuple(observation_params), start, timesteps1))

for i in range(timesteps2):
    start = i
    dyn_obs = dyn_observed.transpose()[start:timesteps2]
    temp = temperature[start:timesteps2]
    l2.append(two_species_lyapunov(dyn_obs, temp, tuple(observation_params), start, timesteps2))


plt.plot(np.arange(timesteps1), l1, color='red')
plt.plot(np.arange(timesteps2), l2, color='blue')
plt.hlines(y = 0, xmin=0, xmax=timesteps1, colors='black', linestyles='--')
plt.ylim(min(l1), 0.05)
plt.xlabel('Time of forecast initialization')
plt.ylabel('Lyapunov')
plt.tight_layout()
plt.savefig(os.path.join(scenario_path, 'lyapunov.pdf'))
#=============#
# Get Lyapunov #
#=============#

observation_params_n = {'growth_rate1', 'CC1', 'competition1', 'temp1','tempsquared1', 'growth_rate2' ,'CC2',  'competition2', 'temp2', 'tempsquared2'}
observation_params = [1.08, 1, 0.021, 0.41, 0.5, 1.06, 1, 0.02, 0.62, 0.72]
d = dict(zip(observation_params_n, observation_params))

dyn_observed, temperature = ft.create_observations(years=1,
                                    observation_params=observation_params,
                                    true_noise=0.0, full_dynamics=True)
def stepwise_increase_parameters(params, target_value=1.7, step=0.05):
    # Use a list comprehension to generate the stepwise increases
    return [[params[0] + i * step, *params[1:5], params[5] + i * step, *params[6:]]
        for i in range(int((target_value - params[0]) / step) + 1)]

# Call the function to get the stepwise increases
increased_observation_params = stepwise_increase_parameters(observation_params, target_value=1.7, step=0.05)

ls = []
for pars in increased_observation_params:

    dyn_observed, temperature = ft.create_observations(years=1,
                                        observation_params=pars,
                                        true_noise=0.0, full_dynamics=True)

    start = 50
    timesteps = 150
    dyn_obs = dyn_observed.transpose()[start:timesteps]
    temp = temperature[start:timesteps]
    ls.append(two_species_lyapunov(dyn_obs, temp, tuple(observation_params), start, timesteps))

plt.plot(ls)