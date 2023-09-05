import torch
import numpy as np
import pandas as pd
import scipy
import os
import os.path
import yaml

from utils import simulate_temperature
from visualisations import plot_losses
from torch.utils.data import DataLoader
from data import SimODEData
from CRPS import CRPS
from scipy.stats import f
import properscoring as ps
from metrics import mse, rolling_corrs, rmse, absolute_differences, fstat, tstat_inverse
from models import Ricker_Predation, Ricker_Ensemble
from sklearn.metrics import r2_score
from loss_functions import crps_loss
from neuralforecast.losses.pytorch import sCRPS, MQLoss, MAE, QuantileLoss
from itertools import product

#===========================================#
# Fit the Ricker model with gradien descent #
#===========================================#

def train(y_train,sigma_train, x_train, model, epochs, loss_fun = 'mse', step_length = 2, fit_sigma = True):


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

            if (loss_fun == 'mse') & (fit_sigma):
                loss = criterion(output, target) + criterion(output_sigma[0], target_upper) + criterion(output_sigma[1], target_lower)
            elif (loss_fun == 'mse') & (not fit_sigma):
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


def save_fit(dictionary, filename, losses, directory_path):

    if not os.path.exists(directory_path):
        # Create the directory if it doesn't exist
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")
    # Convert the dictionary to YAML format
    yaml_data = yaml.dump(dictionary)
    # Write the YAML data to a file
    with open(os.path.join(directory_path, filename), 'w') as file:
        file.write(yaml_data)

    plot_losses(losses, directory_path)


def set_parameters(process = 'stochastic', scenario = 'chaotic'):

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

    return observation_params, initial_params, true_noise, initial_noise
# Create observations
def create_observations(years, observation_params, true_noise):
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

    return y_train, y_test, sigma_train, sigma_test, temp_train, temp_test, climatology

def fit_models(y_train, x_train, sigma_train, initial_params, initial_noise, samples = 20, epochs = 15, loss_fun = 'mse', step_length = 2):

    fitted_values = []

    for i in range(samples):
        # Sample from prior
        ip = [np.random.normal(i, 0.1, 1)[0] for i in initial_params]
        model = Ricker_Ensemble(params=ip, noise=initial_noise, initial_uncertainty=None)
        if initial_noise is not None:
            losses = train(y_train, sigma_train, x_train, model, epochs=epochs, loss_fun = loss_fun, step_length = step_length, fit_sigma=True)
        else:
            losses = train(y_train, sigma_train, x_train, model, epochs=epochs, loss_fun=loss_fun,
                           step_length=step_length, fit_sigma=False)
        print(model.get_fit())
        fm = model.get_fit()
        fitted_values.append(fm)

    return fitted_values, losses

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

def get_forecast_scenario():

    np.random.seed(42)

    rows = ['deterministic', 'stochastic']
    cols = ['chaotic', 'nonchaotic']
    scenarios = list(product(rows, cols))
    obs, preds, ref = [], [], []
    for i in range(len(scenarios)):
        observation_params, initial_params, true_noise, initial_noise = set_parameters(process=scenarios[i][0], scenario=scenarios[i][1])
        y_train, y_test, sigma_train, sigma_test, x_train, x_test, climatology = create_observations(years=30,
                                                                                                     observation_params=observation_params,
                                                                                                     true_noise=true_noise)
        fitted_values = pd.read_csv(f'results/{scenarios[i][1]}_{scenarios[i][0]}/fitted_values.csv')
        fitted_values = fitted_values.drop(fitted_values.columns[0], axis=1)
        parameter_samples = get_parameter_samples(fitted_values,uncertainty=0.02)
        yinit, ypreds, modelfits = forecast_fitted(y_test, x_test, parameter_samples, initial_params, initial_noise,
                                                   initial_uncertainty=0.01)
        obs.append(y_test.detach().numpy())
        preds.append(ypreds)
        ref.append(climatology.detach().numpy())

    return obs, preds, ref

def pointwise_evaluation(forecast, observation, fh_metric, **kwargs):

    if fh_metric == 'crps':
        try:
            performance = [CRPS(forecast[:,i], observation[i]).compute()[0] for i in range(forecast.shape[1])]
        except ValueError:
            performance = [CRPS(forecast[:,i], observation.squeeze()[i]).compute()[0] for i in range(forecast.shape[1])]

    elif fh_metric == 'ae':
        #performance = np.mean(absolute_differences(states[np.newaxis,:], forecast), axis=0)
        performance= np.subtract(observation, forecast)

    elif fh_metric == 'mae':
        #performance = np.mean(absolute_differences(states[np.newaxis,:], forecast), axis=0)
        performance= np.mean(absolute_differences(observation, forecast), axis=0)

    elif fh_metric == 'mse':
        #performance = mse(states[np.newaxis,:], forecast)
        performance = mse(observation, forecast)

    elif fh_metric == 'rmse':
        #performance = rmse(states[np.newaxis,:], forecast)
        performance = rmse(observation, forecast)

    elif fh_metric == 'rsquared':
        performance = [[r2_score(observation[:j], forecast[i,:j]) for i in range(forecast.shape[0])] for j in range(1,forecast.shape[1])]

    elif fh_metric == 'corr':
        w = kwargs['w']
        performance = np.mean(rolling_corrs(observation, forecast, window=w), axis=0)

    elif fh_metric == 'fstats':
        fstats, pvals = fstat(forecast, observation)
        performance = fstats

    return np.array(performance)
def set_threshold(fh_metric, **kwargs):

    if fh_metric == 'mae':
        anomaly = (kwargs['forecast'] - kwargs['observation'] )
        threshold = anomaly.std()
        print('Standard deviation of Anomaly/Residuals: ', threshold)

    elif fh_metric == 'fstats':
        # args[0] = forecast = ensemble size and forecast length
        df1, df2 = kwargs['forecast'].shape[0] - 1, kwargs['forecast'].shape[1] - 1
        threshold = f.ppf(1 - kwargs['alpha']/2, df1, df2)
        print(f"Critical F at alpha = {kwargs['alpha']}:", threshold)

    elif fh_metric == 'corr':
        # args[0] = w = size of moving window
        critical_ts = scipy.stats.t.ppf(1 - kwargs['alpha']/2, kwargs['w'])
        threshold = np.round(tstat_inverse(critical_ts, samples=kwargs['w']), 4)
        print(f"Critical r at alpha = {kwargs['alpha']}:", threshold)

    elif fh_metric == "crps":
        threshold = 0.05
    elif fh_metric == 'mse':
        threshold = 0.025
    elif fh_metric == 'rmse':
        threshold = 0.025

    return threshold

def forecast_skill_horizon(performance, performance_ref, fh_metric):

    skill = performance_ref - performance #1 - (performance/performance_ref)

    if fh_metric == "crps":
        reached_fsh = skill < -0.05
    elif fh_metric == 'mae':
        reached_fsh = skill < -0.05
    elif fh_metric == 'mse':
        reached_fsh = skill < -0.05
    elif fh_metric == 'rmse':
        reached_fsh = skill < -0.05
    elif fh_metric == 'fstats':
        reached_fsh = skill < -0.05

    if reached_fsh.any():
        fsh = np.argmax(reached_fsh)
    else:
        fsh = len(reached_fsh)

    return fsh, skill
def forecast_horizon(performance, fh_metric, threshold):

    if fh_metric != "corr":
        reached_fh = performance > threshold
    else:
        # only correlation is better if larger
        reached_fh = performance < threshold

    if reached_fh.any():
        fh = np.argmax(reached_fh)
    else:
        fh = len(reached_fh)

    return fh

def get_fh(fh_metric, forecast, observation):
    performance = pointwise_evaluation(forecast, observation, fh_metric=fh_metric, w= 5)
    threshold = set_threshold(fh_metric, forecast = forecast, observation = observation, w=5, alpha = 0.05)
    fh = forecast_horizon(performance, fh_metric, threshold)

    return fh

def get_fsh(forecast, reference, obs, fh_metric):
    performance_forecast = pointwise_evaluation(forecast, obs, fh_metric=fh_metric)
    performance_reference = pointwise_evaluation(reference, obs, fh_metric=fh_metric)
    fsh, skill = forecast_skill_horizon(performance_forecast, performance_reference, fh_metric=fh_metric)
    return fsh, skill

