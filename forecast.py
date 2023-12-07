import os
import os.path
import json
import pandas as pd

from data import ForecastData
from torch.utils.data import DataLoader
from CRPS import CRPS
from utils import simulate_temperature
from models import Ricker_Predation, Ricker_Ensemble
from itertools import product

import calibration as ft
import forecast as fc
import numpy as np


from utils import  create_scenario_folder
from visualisations import plot_horizon_maps, plot_scenario_plus_evaluation, plot_all_dynamics, plot_horizons

def set_parameters(process, scenario, dir):

    if scenario == 'chaotic':
        observation_params = [1.18, 1, -0.03, 0.41, 0.25, 1.16, 1, -0.13, 0.62, 0.32] # for chaotic, set r1 = 0.18 and r2 = 0.16 to r1 = 1.08 and r2 = 1.06
        #observation_params = [1.08, 1, 0.021, 0.41, 0.5, 1.06,  1, 0.02, 0.62, 0.72] # for chaotic, set r1 = 0.18 and r2 = 0.16 to r1 = 1.08 and r2 = 1.06
        initial_params = [1.05, 1.0, 0.39, 0.44]
    elif scenario == 'nonchaotic':
        observation_params = [0.18, 1, -0.1, 0.41, 0.25, 0.16, 1, 0.3, 0.32, 0.32]
        # observation_params = [0.18, 1, 0.021, 0.41, 0.5, 0.16, 1, 0.02, 0.62, 0.72]
        initial_params = [0.087132, 0.865437, 0.34348, 0.315365]  # Take mean values from previous runs as prior means.
    if process == 'stochastic':
        true_noise =  0.05# 0.2 # None # [1]
        initial_noise = 0.05  # None # [0.01]
    elif process == 'deterministic':
        true_noise = 0.05 # 0.2
        initial_noise = None  # [0.01]

    parameters = {"observation_params": observation_params,
                  "initial_params": initial_params,
                  "true_noise": true_noise,
                  "initial_noise": initial_noise}
    with open(os.path.join(dir, "parameters.json"), "w") as json_file:
        json.dump(parameters, json_file)

    return parameters
# Create observations
def create_observations(years, observation_params, true_noise, full_dynamics = False):
    timesteps = 365*years
    trainsteps = 365*(years-1)
    teststeps = 365*(1)
    temperature = simulate_temperature(timesteps=timesteps)
    #temperature = standardize(temperature)

    observation_model = Ricker_Predation(params = observation_params, noise = true_noise)
    dyn_observed = observation_model(Temp = temperature)
    y = dyn_observed[0,:].clone().detach().requires_grad_(True)
    y_train, y_test = y[:trainsteps], y[trainsteps:]
    temp_train, temp_test = temperature[:trainsteps], temperature[trainsteps:]
    # Create climatology
    climatology = y_train.view((years-1), 365)
    sigma = np.std(climatology.detach().numpy(), axis=0)
    sigma_train = np.tile(sigma, reps=(years-1))
    sigma_test = sigma

    if full_dynamics:
        return dyn_observed.detach().numpy(), temperature
    else:
        return {'y_train':y_train, 'y_test':y_test,
                'sigma_train':sigma_train, 'sigma_test':sigma_test,
                'x_train':temp_train, 'x_test':temp_test,
                'climatology':climatology}


def forecast_fitted(y_test, x_test, fitted_values, initial_params, initial_noise, initial_uncertainty = None):
    yinit = []
    ypreds = []
    modelfits = []
    for fm in fitted_values:
        params = [v for v in fm.values()][:4]
        modelinit = Ricker_Ensemble(params = initial_params, noise=initial_noise, initial_uncertainty=None)
        modelfit = Ricker_Ensemble(params = params, noise=fm['sigma'], initial_uncertainty=None)
        modelfits.append(modelfit)
        if initial_uncertainty is not None:
            Ninit = np.random.normal(y_test[0].detach().numpy(), y_test[0].detach().numpy()*initial_uncertainty, 10)
            yinit.append(np.array([modelinit.forecast(N0=Ninit[i], Temp=x_test).detach().numpy() for i in range(len(Ninit))]))
            ypreds.append(np.array([modelfit.forecast(N0=Ninit[i], Temp=x_test).detach().numpy() for i in range(len(Ninit))]))
        else:
            yinit.append(modelinit.forecast(N0=y_test[0], Temp=x_test).detach().numpy())
            ypreds.append(modelfit.forecast(N0=y_test[0], Temp=x_test).detach().numpy())
    if initial_uncertainty is not None:
        try:
            yinit = np.stack(yinit).squeeze()
            yinit = yinit.reshape(-1, yinit.shape[2])
            ypreds = np.stack(ypreds).squeeze()
            ypreds = ypreds.reshape(-1, ypreds.shape[2])
        except IndexError:
            yinit = np.array(yinit).squeeze()
            ypreds = np.array(ypreds).squeeze()
    else:
        yinit = np.array(yinit).squeeze()
        ypreds = np.array(ypreds).squeeze()
    ypreds = ypreds[~np.any(np.isnan(ypreds), axis=1), :]
    return yinit, ypreds, modelfits

def get_parameter_samples(fitted_values, uncertainty = 0.05):

    fitted_pars = fitted_values.mean().values
    # allow variation of 5 %
    ip = np.array([np.random.normal(i, i*uncertainty, 50) for i in fitted_pars])
    keys = ['alpha', 'beta', 'bx', 'cx', 'sigma', 'phi']
    ip_samples = [dict(zip(keys, ip[:,column])) for column in range(ip.shape[1])]
    ip_samples = [{key: None if isinstance(value, float) and np.isnan(value) else value for key, value in sample.items()} for sample in ip_samples]

    return ip_samples

def get_forecast_scenario(dir):

    np.random.seed(42)

    rows = ['deterministic', 'stochastic']
    cols = ['chaotic', 'nonchaotic']
    scenarios = list(product(rows, cols))
    obs, preds, ref = {},{},{}
    for i in range(len(scenarios)):
        parameters = set_parameters(process=scenarios[i][0], scenario=scenarios[i][1], dir=dir)
        observations = create_observations(years=30, observation_params=parameters['observation_params'],
                                          true_noise=parameters['true_noise'])
        fitted_values = pd.read_csv(os.path.join(dir, f'{scenarios[i][0]}_{scenarios[i][1]}/fitted_values.csv'))
        fitted_values = fitted_values.drop(fitted_values.columns[0], axis=1)
        parameter_samples = get_parameter_samples(fitted_values,uncertainty=0.02)
        yinit, predictions, modelfits = forecast_fitted(observations['y_test'], observations['x_test'],
                                                    parameter_samples,
                                                    parameters['initial_params'], parameters['initial_noise'],
                                                   initial_uncertainty=0.01)

        obs[f'{scenarios[i][1]}_{scenarios[i][0]}'] = observations['y_test'].detach().numpy()
        preds[f'{scenarios[i][1]}_{scenarios[i][0]}'] = predictions
        ref[f'{scenarios[i][1]}_{scenarios[i][0]}'] = observations['climatology'].detach().numpy()

    return obs, preds, ref



def forecast_different_leads(y_test, x_test, climatology, modelfits, forecast_days = 110, lead_time = 110):

    data = ForecastData(y_test, x_test, climatology, forecast_days=forecast_days, lead_time=lead_time)
    forecastloader = DataLoader(data, batch_size=1, shuffle=False, drop_last=True)

    mat_ricker = np.full((lead_time, forecast_days), np.nan)
    mat_ricker_perfect = np.full((lead_time, forecast_days), np.nan)
    mat_climatology = np.full((lead_time, forecast_days), np.nan)
    mat_climatology_perfect = np.full((lead_time, forecast_days), np.nan)

    i = 0
    fh_metric = 'crps'
    for states, temps, clim in forecastloader:

        print('I is: ', i)
        N0 = states[:, 0]
        clim = clim.squeeze().detach().numpy()
        forecast = []
        for modelfit in modelfits:
            forecast.append(modelfit.forecast(N0, temps).detach().numpy())
        forecast = np.array(forecast).squeeze()
        states = states.squeeze().detach().numpy()

        if fh_metric == 'crps':
            performance = [CRPS(forecast[:, i], states[i]).compute()[0] for i in range(forecast.shape[1])]
            performance_ref = [CRPS(clim[:, i], states[i]).compute()[0] for i in range(clim.shape[1])]
            mat_ricker[:, i] = performance
            mat_climatology[:, i] = performance_ref

            performance_perfect = [CRPS(forecast[:, i], forecast[:, i].mean(axis=0)).compute()[0] for i in
                                   range(forecast.shape[1])]
            performance_climatology_perfect = [CRPS(clim[:, i], forecast[:, i].mean(axis=0)).compute()[0] for i in
                                               range(forecast.shape[1])]
            mat_ricker_perfect[:, i] = performance_perfect
            mat_climatology_perfect[:, i] = performance_climatology_perfect

        i += 1

    return mat_ricker, mat_climatology, mat_ricker_perfect


def create_experimental_data(experiment_path, process, scenario, fit_model=False):

    scenario_path = create_scenario_folder(experiment_path, new_folder_name = f'{process}_{scenario}')

    parameters = fc.set_parameters(process=process, scenario=scenario,dir = scenario_path)

    observations = fc.create_observations(years=30,observation_params=parameters['observation_params'],
                                          true_noise=parameters['true_noise'])

    # get calibrated parameters
    fitted_values = ft.model_fit(fit_model, scenario_path,
                                 observations['y_train'], observations['x_train'], observations['sigma_train'],
                                 parameters['initial_params'], parameters['initial_noise'],
                                 samples=1, epochs=15, loss_fun='mse', step_length=10)

    # Forecast with the one scenario #
    parameter_samples = fc.get_parameter_samples(fitted_values, uncertainty=0.02)
    yinit, forecast, modelfits = fc.forecast_fitted(observations['y_test'], observations['x_test'],
                                                    parameter_samples,
                                                    parameters['initial_params'], parameters['initial_noise'],
                                                    initial_uncertainty=0.01)

    plot_scenario_plus_evaluation(forecast, observations['y_test'], dir = scenario_path, clim=observations['climatology'])

    # Forecasting with the fitted at different lead times #
    mat_ricker, mat_climatology, mat_ricker_perfect = fc.forecast_different_leads(observations['y_test'], observations['x_test'],
                                                                                      observations['climatology'], modelfits,
                                                                                   forecast_days = 110, lead_time = 110)

    plot_horizon_maps(mat_ricker, mat_climatology, mat_ricker_perfect, dir = scenario_path)
