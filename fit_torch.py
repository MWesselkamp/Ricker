import torch
import numpy as np
import pandas as pd

import os
import os.path
import json

from utils import simulate_temperature
from visualisations import plot_losses, plot_posterior
from torch.utils.data import DataLoader
from data import SimODEData
from models import Ricker_Predation, Ricker_Ensemble
from loss_functions import crps_loss
from neuralforecast.losses.pytorch import sCRPS, MQLoss, MAE, QuantileLoss
from itertools import product

#===========================================#
# Fit the Ricker model with gradien descent #
#===========================================#

def train(y_train,sigma_train, x_train, model, epochs, loss_fun = 'mse', step_length = 2, fit_sigma = None):


    data = SimODEData(step_length=step_length, y=y_train, y_sigma=sigma_train, temp=x_train)
    trainloader = DataLoader(data, batch_size=1, shuffle=False, drop_last=True)

    optimizer = torch.optim.Adam([{'params':model.model_params}], lr=0.0001)

    criterion = torch.nn.MSELoss()
    criterion2 = sCRPS()
    criterion3 = MQLoss(quantiles = [0.4, 0.6])
    criterion4 = QuantileLoss(0.5) # Pinball Loss
    criterion5 = torch.nn.GaussianNLLLoss()

    losses = []
    for epoch in range(epochs):

        if epoch % 10 == 0:
            print('Epoch:', epoch)

        for batch in trainloader:

            target, var, temp = batch
            target = target.squeeze()

            target_upper = target + 2*torch.std(target).item()
            target_lower = target - 2*torch.std(target).item()

            initial_state = target.clone()[0]

            optimizer.zero_grad()

            output, output_sigma = model(initial_state, temp)

            if (loss_fun == 'mse') & (fit_sigma is not None):
                loss = criterion(output, target) + criterion(output_sigma[0], target_upper) + criterion(output_sigma[1], target_lower)
            elif (loss_fun == 'mse') & (fit_sigma is None):
                loss = criterion(output, target)
            elif loss_fun == 'crps':
                # loss = torch.zeros((1), requires_grad=True).clone()
                loss = torch.stack([crps_loss(output[:,i].squeeze(), target[i]) for i in range(step_length)])
            elif loss_fun == 'quantile':
                loss = torch.stack([criterion4(target[i], output[:,i].squeeze()) for i in range(step_length)])
            elif loss_fun == 'mquantile':
                pass
            elif loss_fun == 'gaussian':
                loss = criterion5(output, target, output_sigma)

            loss = torch.sum(loss) / step_length

            loss.backward()
            losses.append(loss.clone())
            optimizer.step()

    return losses


def model_fit(fit_model, dir, y_train, x_train, sigma_train, initial_params, initial_noise, **kwargs):

    if ('fitted_values' not in globals()) & (fit_model):

        fitted_values = []
        for i in range(kwargs['samples']):
            # Sample from prior
            ip = [np.random.normal(i, 0.1, 1)[0] for i in initial_params]
            model = Ricker_Ensemble(params=ip, noise=initial_noise, initial_uncertainty=None)

            losses = train(y_train, sigma_train, x_train, model,
                           epochs=kwargs['epochs'], loss_fun=kwargs['loss_fun'], step_length=kwargs['step_length'],
                           fit_sigma=initial_noise)

            fitted_values.append(model.get_fit())
            print(model.get_fit())

        fitted_values = pd.DataFrame(fitted_values)
        fitted_values.to_csv(os.path.join(dir, 'fitted_values.csv'))

        plot_posterior(fitted_values, saveto=dir)
        plot_losses(losses, loss_fun='mse', saveto=dir)

    else:
        print(f"Loading parameters from previous fit.")
        fitted_values = pd.read_csv(os.path.join(dir, 'fitted_values.csv'), index_col=False)
        fitted_values = fitted_values.drop(fitted_values.columns[0], axis=1)

    return fitted_values


def set_parameters(process, scenario, dir):

    if scenario == 'chaotic':
        observation_params = [1.08, 1, 0.021, 0.41, 0.5, 1.06,  1, 0.02, 0.62, 0.72] # for chaotic, set r1 = 0.18 and r2 = 0.16 to r1 = 1.08 and r2 = 1.06
        initial_params = [0.95, 1.0, 0.39, 0.44]
    elif scenario == 'nonchaotic':
        observation_params = [0.18, 1, 0.021, 0.41, 0.5, 0.16, 1, 0.02, 0.62, 0.72]
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

    return observation_params, initial_params, true_noise, initial_noise
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
        return y_train, y_test, sigma_train, sigma_test, temp_train, temp_test, climatology


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
    obs, preds, ref = [], [], []
    for i in range(len(scenarios)):
        observation_params, initial_params, true_noise, initial_noise = set_parameters(process=scenarios[i][0], scenario=scenarios[i][1], dir=dir)
        y_train, y_test, sigma_train, sigma_test, x_train, x_test, climatology = create_observations(years=30,
                                                                                                     observation_params=observation_params,
                                                                                                     true_noise=true_noise)
        fitted_values = pd.read_csv(os.path.join(dir, f'{scenarios[i][1]}_{scenarios[i][0]}/fitted_values.csv'))
        fitted_values = fitted_values.drop(fitted_values.columns[0], axis=1)
        parameter_samples = get_parameter_samples(fitted_values,uncertainty=0.02)
        yinit, ypreds, modelfits = forecast_fitted(y_test, x_test, parameter_samples, initial_params, initial_noise,
                                                   initial_uncertainty=0.01)
        obs.append(y_test.detach().numpy())
        preds.append(ypreds)
        ref.append(climatology.detach().numpy())

    return obs, preds, ref

