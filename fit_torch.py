import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import argparse
import os
import os.path
import yaml


from utils import simulate_temperature, standardize
from visualisations import plot_losses, plot_posterior, plot_fit, plot_fit2, plot_all_dynamics, plot_horizons
from torch.utils.data import DataLoader
from data import SimODEData, ForecastData
from CRPS import CRPS
from scipy.stats import f, bartlett
import properscoring as ps
from metrics import rolling_mse, mse, rolling_corrs, rmse, absolute_differences
from sklearn.metrics import r2_score
from loss_functions import crps_loss
from neuralforecast.losses.pytorch import sCRPS, MQLoss, MAE, QuantileLoss
from itertools import product

#parse = False
#set flags
#if parse:
#
#    parser = argparse.ArgumentParser()
#    parser.add_argument("--process", type=str, help="Set process type to stochastic or deterministic")
#    parser.add_argument("--scenario", type=str, help="Set scenario type to chaotic or nonchaotic")
#    parser.add_argument("--loss_fun", type=str, help="Set loss function to quantile, crps or mse")
#    parser.add_argument("--fh_metric", type=str, help="Set horizon metric to crps, mse or nashsutcliffe")

    # Parse the command-line arguments
#    args = parser.parse_args()
#
#    process = args.process
#    scenario = args.scenario
#    loss_fun = args.loss_fun
#    fh_metric = args.fh_metric

#===========================================#
# Fit the Ricker model with gradien descent #
#===========================================#

class Ricker_Predation(nn.Module):
    """

    """

    def __init__(self, params, noise = None):
        super().__init__()
        if not noise is None:
            self.model_params = torch.nn.Parameter(torch.tensor(params + [noise], requires_grad=True, dtype=torch.double))
            self.noise = noise
        else:
            self.model_params = torch.nn.Parameter(torch.tensor(params, requires_grad=True, dtype=torch.double))
            self.noise = noise

    def forward(self, Temp):

        if not self.noise is None:
            alpha1, beta1, gamma1, bx1, cx1, alpha2, beta2, gamma2, bx2, cx2, sigma = self.model_params
        else:
            alpha1, beta1, gamma1, bx1, cx1, alpha2, beta2, gamma2, bx2, cx2 = self.model_params

        Temp = Temp.squeeze()
        out = torch.ones((2,len(Temp)), dtype=torch.double)

        if not self.noise is None:
            for i in range(len(Temp) - 1):
                out[0, i + 1] = out.clone()[0, i] * torch.exp(alpha1*(1 - beta1*out.clone()[0, i] - gamma1*out.clone()[1, i] + bx1 * Temp[i] + cx1 * Temp[i]**2)) \
                                + sigma*torch.normal(mean=torch.tensor([0.0,]), std=torch.tensor([0.1]))
                out[1, i + 1] = out.clone()[1, i] * torch.exp(alpha2*(1 - beta2*out.clone()[1, i] - gamma2*out.clone()[0, i] + bx2 * Temp[i] + cx2 * Temp[i]**2)) \
                                + sigma*torch.normal(mean=torch.tensor([0.0,]), std=torch.tensor([0.1]))
        else:
            for i in range(len(Temp) - 1):
                out[0, i + 1] = out.clone()[0, i] * torch.exp(alpha1*(1 - beta1*out.clone()[0, i] - gamma1*out.clone()[1, i] + bx1 * Temp[i] + cx1 * Temp[i]**2))
                out[1, i + 1] = out.clone()[1, i] * torch.exp(alpha2*(1 - beta2*out.clone()[1, i] - gamma2*out.clone()[0, i] + bx2 * Temp[i] + cx2 * Temp[i]**2))

        return out

    def __repr__(self):
        return f" alpha1: {self.model_params[0].item()}, \
            beta1: {self.model_params[1].item()}, \
                gamma1: {self.model_params[2].item()}, \
        bx1: {self.model_params[3].item()}, \
                    cx1: {self.model_params[4].item()}, \
                        alpha2: {self.model_params[5].item()}, \
        beta2: {self.model_params[6].item()}, \
                    gamma2: {self.model_params[7].item()}, \
                        bx2: {self.model_params[8].item()}, \
                    cx2: {self.model_params[9].item()},\
               sigma:{self.noise} "


class Ricker(nn.Module):
    """
     Single species extended Ricker Model
    """

    def __init__(self, params, noise=None):

        super().__init__()

        if not noise is None:
            self.model_params = torch.nn.Parameter(torch.tensor(params+noise, requires_grad=True, dtype=torch.double))
            self.noise = noise
        else:
            self.model_params = torch.nn.Parameter(torch.tensor(params, requires_grad=True, dtype=torch.double))
            self.noise = None

    def forward(self, N0, Temp):

        if not self.noise is None:
            alpha, beta, bx, cx, sigma = self.model_params
        else:
            alpha, beta, bx, cx = self.model_params

        Temp = Temp.squeeze()

        out = torch.zeros_like(Temp, dtype=torch.double)
        out[0] = N0 # initial value

        if not self.noise is None:
            for i in range(len(Temp)-1):
                out[i+1] = out.clone()[i] * torch.exp(alpha * (1 - beta * out.clone()[i] + bx * Temp[i] + cx * Temp[i] ** 2)) \
                           + sigma*torch.normal(mean=torch.tensor([0.0,]), std=torch.tensor([1.0]))
        else:
            for i in range(len(Temp)-1):
                out[i+1] = out.clone()[i] * torch.exp(alpha * (1 - beta * out.clone()[i] + bx * Temp[i] + cx * Temp[i] ** 2))

        return out

    def __repr__(self):
        return f" alpha: {self.model_params[0].item()}, \
            beta: {self.model_params[1].item()}, \
                bx: {self.model_params[2].item()}, \
                    cx: {self.model_params[3].item()}, \
               sigma: {self.noise}"


class Ricker_Ensemble(nn.Module):
    """
     Single-Species extended Ricker with Ensemble prediction.
    """

    def __init__(self, params, noise=None, initial_uncertainty = None):

        super().__init__()

        if (not noise is None) & (not initial_uncertainty is None):
            self.model_params = torch.nn.Parameter(torch.tensor(params + [noise] + [initial_uncertainty], requires_grad=True, dtype=torch.double))
            self.initial_uncertainty = initial_uncertainty
            self.noise = noise
        elif (not noise is None) & (initial_uncertainty is None):
            self.model_params = torch.nn.Parameter(torch.tensor(params + [noise], requires_grad=True, dtype=torch.double))
            self.initial_uncertainty = initial_uncertainty
            self.noise = noise
        elif (noise is None) & (not initial_uncertainty is None):
            self.model_params = torch.nn.Parameter(torch.tensor(params + [initial_uncertainty], requires_grad=True, dtype=torch.double))
            self.initial_uncertainty = initial_uncertainty
            self.noise = noise
        elif (noise is None) & (initial_uncertainty is None):
            self.model_params = torch.nn.Parameter(torch.tensor(params, requires_grad=True, dtype=torch.double))
            self.noise = noise
            self.initial_uncertainty = initial_uncertainty

    def forward(self, N0, Temp, ensemble_size=15):

        if (not self.noise is None) & (not self.initial_uncertainty is None):
            alpha, beta, bx, cx, sigma, phi = self.model_params
        elif (not self.noise is None) & (self.initial_uncertainty is None):
            alpha, beta, bx, cx, sigma = self.model_params
        elif (self.noise is None) & (not self.initial_uncertainty is None):
            alpha, beta, bx, cx, phi = self.model_params
        else:
            alpha, beta, bx, cx = self.model_params

        Temp = Temp.squeeze()

        if not self.initial_uncertainty is None:
            initial = N0 + phi * torch.normal(torch.zeros((ensemble_size)), torch.repeat_interleave(torch.tensor([.1, ]), ensemble_size))
            out = torch.zeros((len(initial), len(Temp)), dtype=torch.double)
        else:
            initial = N0
            out = torch.zeros((1, len(Temp)), dtype=torch.double)

        out[:,0] = initial  # initial value

        if not self.noise is None:
            for i in range(len(Temp) - 1):
                out[:,i + 1] = out.clone()[:,i] * torch.exp(
                    alpha * (1 - beta * out.clone()[:,i] + bx * Temp[i] + cx * Temp[i] ** 2))# \
                          #   + sigma * torch.normal(mean=torch.tensor([0.0, ]), std=torch.tensor([1.0, ]))
            var = sigma * 2#torch.normal(mean=torch.tensor([0.0, ]), std=torch.tensor([1.0, ]))
            out_upper = out + torch.repeat_interleave(var, len(Temp)) #+ torch.full_like(out, var.item())
            out_lower = out - torch.repeat_interleave(var, len(Temp))  #- torch.full_like(out, var.item())

            return out, [out_upper, out_lower]

        else:
            for i in range(len(Temp) - 1):
                out[:,i + 1] = out.clone()[:,i] * torch.exp(
                    alpha * (1 - beta * out.clone()[:,i] + bx * Temp[i] + cx * Temp[i] ** 2))

            return out, None

    def get_fit(self):

        return {"alpha": self.model_params[0].item(), \
            "beta": self.model_params[1].item(), \
                "bx": self.model_params[2].item(), \
                    "cx": self.model_params[3].item(), \
               "sigma": self.noise if self.noise is None else self.model_params[4].item(), \
               "phi": self.initial_uncertainty if self.initial_uncertainty is None else (self.model_params[5].item() if self.noise is not None else self.model_params[4].item())
                }

    def forecast(self, N0, Temp, ensemble_size=15):

        if (not self.noise is None) & (not self.initial_uncertainty is None):
            alpha, beta, bx, cx, sigma, phi = self.model_params
        elif (not self.noise is None) & (self.initial_uncertainty is None):
            alpha, beta, bx, cx, sigma = self.model_params
        elif (self.noise is None) & (not self.initial_uncertainty is None):
            alpha, beta, bx, cx, phi = self.model_params
        else:
            alpha, beta, bx, cx = self.model_params

        Temp = Temp.squeeze()

        if not self.initial_uncertainty is None:
            initial = N0 + phi * torch.normal(torch.zeros((ensemble_size)),
                                              torch.repeat_interleave(torch.tensor([.1, ]), ensemble_size))
            out = torch.zeros((len(initial), len(Temp)), dtype=torch.double)
        else:
            initial = N0
            out = torch.zeros((1, len(Temp)), dtype=torch.double)

        out[:, 0] = initial  # initial value

        if not self.noise is None:
            for i in range(len(Temp) - 1):
                out[:, i + 1] = out.clone()[:, i] * torch.exp(
                    alpha * (1 - beta * out.clone()[:, i] + bx * Temp[i] + cx * Temp[i] ** 2))  \
                    + sigma * torch.normal(mean=torch.tensor([0.0, ]), std=torch.tensor([0.1, ]))

            #out = out + sigma * torch.normal(mean=torch.tensor([0.0, ]), std=torch.tensor([1.0, ]))

            return out

        else:
            for i in range(len(Temp) - 1):
                out[:, i + 1] = out.clone()[:, i] * torch.exp(
                    alpha * (1 - beta * out.clone()[:, i] + bx * Temp[i] + cx * Temp[i] ** 2))

            return out

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
        true_noise =  0.2 # None # [1]
        initial_noise = 0.05  # None # [0.01]
    elif process == 'deterministic':
        true_noise = 0.2
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

    plt.plot(y_test.detach().numpy())
    plt.show()
    plt.close()

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
    for fm in fitted_values:
        params = [v for v in fm.values()][:4]
        modelinit = Ricker_Ensemble(params = initial_params, noise=initial_noise, initial_uncertainty=None)
        modelfit = Ricker_Ensemble(params = params, noise=fm['sigma'], initial_uncertainty=None)
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
    return yinit, ypreds

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
        yinit, ypreds = forecast_fitted(y_test, x_test, parameter_samples, initial_params, initial_noise,
                                                   initial_uncertainty=0.01)
        obs.append(y_test.detach().numpy())
        preds.append(ypreds)
        ref.append(climatology.detach().numpy())

    return obs, preds, ref

def fstat(forecast, observation, bartlett_test = False):

    if bartlett_test:
        bs = np.array([bartlett(forecast[:, i].squeeze(), observation[:,:150].squeeze()) for i in range(forecast.shape[1])])
        stats = bs[:,0]
        pvals = bs[:,1]
    else:
        vars = [np.var(forecast[:, i]) for i in range(forecast.shape[1])]
        var_obs = np.var(observation)
        stats = [(vars[i] / var_obs) for i in range(len(vars))] # ppp = [(1-vars[i]/var_true) for i in range(len(vars))]
        df1, df2 = forecast.shape[0] - 1, forecast.shape[1] - 1
        pvals = [f.cdf(stats[i], df1, df2) for i in range(len(stats))]
    return stats, pvals

def pointwise_evaluation(forecast, observation, fh_metric):

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
        w = 10
        performance = np.mean(rolling_corrs(observation, forecast, window=w), axis=0)

    elif fh_metric == 'fstats':
        fstats, pvals = fstat(forecast, observation)
        performance = fstats

    return np.array(performance)

def forecast_skill_horizon(performance, performance_ref, fh_metric):

    ratio = (performance/performance_ref)
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
def forecast_horizon(performance, fh_metric, *args):

    if fh_metric == "crps":
        reached_fh = performance > 0.05
    elif fh_metric == 'mae':
        reached_fh = performance > 0.14 # anomaly_true.std() 0.14770598608820823
    elif fh_metric == 'mse':
        reached_fh = performance > 0.025
    elif fh_metric == 'rmse':
        reached_fh = performance > 0.025
    elif fh_metric == 'corr':
        reached_fh = performance < 0.5
    elif fh_metric == 'fstats':
        df1, df2 = args[0].shape[0] - 1, args[0].shape[1] - 1
        alpha = 0.05
        critical_F = f.ppf(1 - alpha / 2, df1, df2)
        reached_fh = performance > critical_F

    if reached_fh.any():
        fh = np.argmax(reached_fh)
    else:
        fh = len(reached_fh)

    return fh
def get_fh(forecast, obs, fh_metric):
    performance = pointwise_evaluation(forecast, obs, fh_metric=fh_metric)
    if fh_metric == 'fstats':
        fh = forecast_horizon(performance, fh_metric, forecast)
    else:
        fh = forecast_horizon(performance, fh_metric)

    return fh

def get_fsh(forecast, reference, obs, fh_metric):
    performance_forecast = pointwise_evaluation(forecast, obs, fh_metric=fh_metric)
    performance_reference = pointwise_evaluation(reference, obs, fh_metric=fh_metric)
    fsh, skill = forecast_skill_horizon(performance_forecast, performance_reference, fh_metric=fh_metric)
    return fsh, skill

#========================#
# Set simulation setting #
#========================#

plt.rcParams['font.size'] = 18

fit = False
compute_horizons = False
seed = True

process = 'stochastic'
scenario = 'chaotic'

if seed:
    np.random.seed(42)

observation_params, initial_params, true_noise, initial_noise = set_parameters(process = process, scenario = scenario)
y_train, y_test, sigma_train, sigma_test, x_train, x_test, climatology = create_observations(years = 30,
                                                                                             observation_params= observation_params,
                                                                                             true_noise = true_noise)

#============#
# Fit Model  #
#============#

if fit:

    if process == 'stochastic':
        fitted_values, losses = fit_models(y_train, x_train, sigma_train, initial_params, initial_noise, samples=1, epochs=20, loss_fun = 'mse', step_length = 10)
    else:
        fitted_values, losses = fit_models(y_train, x_train, sigma_train, initial_params, initial_noise, samples=1, epochs=20, loss_fun = 'mse', step_length = 20)

    fitted_values = pd.DataFrame(fitted_values)

    fitted_values.to_csv(f'results/{scenario}_{process}/fitted_values.csv')
    plot_posterior(fitted_values, saveto=f'results/{scenario}_{process}')
    plot_losses(losses, loss_fun='mse', saveto=f'results/{scenario}_{process}')

#=======================#
# Forecast all dynamics #
#=======================#


obs, preds, ref = get_forecast_scenario()
plot_all_dynamics(obs, preds, ref, save=True)

#================================#
# Forecast with the one scenario #
#================================#

if 'fitted_values' in globals():
    print(f"Deleting globally set parameter values.")
    del(fitted_values)
    print(f"Loading parameters from previous fit.")
    fitted_values = pd.read_csv(f'results/{scenario}_{process}/fitted_values.csv', index_col=False)
    fitted_values = fitted_values.drop(fitted_values.columns[0], axis=1)
else:
    print(f"Loading parameters from previous fit.")
    fitted_values = pd.read_csv(f'results/{scenario}_{process}/fitted_values.csv', index_col=False)
    fitted_values = fitted_values.drop(fitted_values.columns[0], axis=1)

parameter_samples = get_parameter_samples(fitted_values, uncertainty=0.02)
yinit, forecast = forecast_fitted(y_test, x_test, parameter_samples, initial_params, initial_noise, initial_uncertainty = 0.01)

plot_fit(forecast, y_test, scenario=f"{scenario}", process=f"{process}", clim=climatology,save=True)
plot_fit2(forecast, y_test,scenario=f"{scenario}", process=f"{process}", clim=climatology,save=True)
#sr = [shapiro(ypreds[:,i])[1] for i in range(ypreds.shape[1])]
#plt.plot(sr)

#fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(6, 8), sharex=True)
#for i in range(7):
#    x = ypreds[:,i]
#    mu = x.mean()
#    sigma = x.std()
#    axes[i].hist(x=climatology[:, i].detach().numpy(),bins=20, alpha=0.5, colors='salmon')
#    axes[i].hist(x, bins=20, alpha=0.5)
#    xs = np.linspace(x.min(), x.max(), num=100)
#    axes[i].plot(xs, stats.norm.pdf(xs, mu, sigma))
    #axes[i].vlines(x = y_test[i].detach().numpy(), ymin = 0, ymax = 50)

#for i in range(5):
#    print('Ensemble', ps.crps_ensemble(y_test[i].detach().numpy(), ypreds[:,i]))
#    print('Climatology', ps.crps_ensemble(y_test[i].detach().numpy(), climatology.detach().numpy()[:,i]))

#=============================#
# Forecasting with the fitted #
#=============================#



observation = y_test.detach().numpy()[np.newaxis,:]
reference = climatology.detach().numpy()
reference2 = np.tile(reference[:,-1], (reference.shape[1],1)).transpose()
obs_perfect = np.mean(forecast, axis=0)[np.newaxis,:]
ref_perfect = np.mean(reference, axis=0)[np.newaxis,:]

anomaly_clim = (forecast - ref_perfect)
anomaly_true = (forecast - observation)
print('Standard deviation of Residuals: ', anomaly_true.std())

#perf = pointwise_evaluation(forecast, obs, fh_metric='fstats')
#performanceref = pointwise_evaluation(reference, obs, fh_metric='mae')
#fh = get_fh(forecast, obs_perfect, fh_metric='crps')
#fsh, skill = get_fsh(forecast, reference, obs_perfect, fh_metric='crps')
#plt.plot(performance)
#plt.plot(performanceref)
#from scipy.stats import norm
#mse_pvals = [norm.cdf(skill[i], loc=0, scale=1) for i in range(skill.shape[0])]

#crps_mod = np.array([ps.crps_ensemble(obs[:,i].item(), forecast[:,i]) for i in range(forecast.shape[1])])
#crps_ref = np.array([ps.crps_ensemble(obs[:,i].item(), reference[:,i]) for i in range(forecast.shape[1])])
#crps_ref2 = np.array([ps.crps_ensemble(obs[:,i].item(), reference2[:,i]) for i in range(forecast.shape[1])])
#plt.plot(crps_mod)
#plt.plot(crps_ref)
#plt.plot(crps_ref2)
#plt.plot((performanceref - performance))
#np.argmax((performanceref - performance) < -0.02)

if compute_horizons:
    print(f'Computing forecast horizons for {scenario}_{process} setting')
    metrics_fh = ['corr', 'mae', 'fstats', 'crps'] # ['corr', 'mse', 'mae', 'crps']

    fha_ricker = [get_fh(forecast, obs, fh_metric=m) for m in metrics_fh]
    fhp_ricker = [get_fh(forecast, obs_perfect, fh_metric=m) for m in metrics_fh]
    fha_reference = [get_fh(reference, obs, fh_metric=m) for m in metrics_fh]
    fhp_reference = [get_fh(reference, ref_perfect, fh_metric=m) for m in metrics_fh]

    metrics_fsh = ['crps'] # ['mse', 'mae', 'crps']
    fsh = [None, None, None] + [get_fsh(forecast, reference,obs, fh_metric=m)[0] for m in metrics_fsh]
    fsh2 = [None, None, None] + [get_fsh(forecast, reference2,obs, fh_metric=m)[0] for m in metrics_fsh]


    fhs = pd.DataFrame([fha_ricker, fhp_ricker, fha_reference, fhp_reference, fsh], columns=metrics_fh,
                 index = ['fha_ricker', 'fhp_ricker', 'fha_reference', 'fhp_reference', 'fsh'])
    fhs.to_csv(f'results/{scenario}_{process}/horizons.csv')
else:
    print(f'LOADING forecast horizons from previous run of {scenario}_{process} setting')
    fhs = pd.read_csv(f'results/{scenario}_{process}/horizons.csv', index_col=0)

plot_fit2(forecast, y_test,scenario=f"{scenario}", process=f"{process}",
          clim=climatology,fhs=[fhs['fstats']['fha_ricker'], fhs['crps']['fsh']], save=True)

plot_horizons(fha_ricker,fha_reference, fhp_ricker, fhp_reference, fsh, metrics_fh, scenario, process, save=True)


#================================================#
# Forecasting with the fitted at same lead times #
#================================================#

forecast_days = 110
lead_time = 110
data = ForecastData(y_test,x_test, climatology, forecast_days=forecast_days, lead_time=lead_time)
forecastloader = DataLoader(data, batch_size=1, shuffle=False, drop_last=True)

mat_ricker = np.full((lead_time, forecast_days), np.nan)
mat_ricker_perfect = np.full((lead_time, forecast_days), np.nan)
mat_climatology = np.full((lead_time, forecast_days), np.nan)
mat_climatology_perfect = np.full((lead_time, forecast_days), np.nan)

i = 0
fh_metric = 'crps'
for states, temps, clim in forecastloader:

    print('I is: ', i)
    N0 = states[:,0]
    clim = clim.squeeze().detach().numpy()
    forecast = []
    for modelfit in modelfits:
        forecast.append(modelfit.forecast(N0, temps).detach().numpy())
    forecast = np.array(forecast).squeeze()
    states = states.squeeze().detach().numpy()

    if fh_metric == 'crps':
        performance = [CRPS(forecast[:,i], states[i]).compute()[0] for i in range(forecast.shape[1])]
        performance_ref = [CRPS(clim[:, i], states[i]).compute()[0] for i in range(clim.shape[1])]
        mat_ricker[:, i] = performance
        mat_climatology[:, i] = performance_ref

        performance_perfect = [CRPS(forecast[:, i], forecast[:,i].mean(axis=0)).compute()[0] for i in
                               range(forecast.shape[1])]
        performance_climatology_perfect = [CRPS(clim[:, i], forecast[:, i].mean(axis=0)).compute()[0] for i in
                               range(forecast.shape[1])]
        mat_ricker_perfect[:, i] = performance_perfect
        mat_climatology_perfect[:, i] = performance_climatology_perfect

    i+=1

fig, ax = plt.subplots(2, 3, figsize=(14, 9), sharey=False, sharex=False)
custom_min = 0
custom_max = 0.6

heatmap0 = ax[0,0].imshow(mat_ricker.transpose()[0:,:], vmin=custom_min, vmax=custom_max, cmap='plasma_r')
cb = plt.colorbar(heatmap0, ax=ax[0,0])
cb.set_label('CRPS')
ax[0,0].set_ylabel('Day of forecast initialization')
ax[0,0].set_xlabel('Forecast time')
ax[0,0].autoscale(False)

ax[0,1].imshow((mat_ricker.transpose()[0:,:] > 0.05), cmap='plasma_r')
#plt.colorbar()
ax[0,1].set_ylabel('Day of forecast initialization')
ax[0,1].set_xlabel('Forecast time')

ax[0,2].imshow((mat_ricker.transpose()[0:,:] > mat_climatology.transpose()[0:,:]), cmap='plasma_r')
#ax[0,2].colorbar()
ax[0,2].set_ylabel('Day of forecast initialization')
ax[0,2].set_xlabel('Forecast time')

heatmap1 = ax[1,0].imshow(mat_ricker_perfect.transpose()[0:,:], vmin=custom_min, vmax=custom_max, cmap='plasma_r')
cb1 = plt.colorbar(heatmap1, ax=ax[1,0])
cb1.set_label('CRPS')
ax[1,0].set_ylabel('Day of forecast initialization')
ax[1,0].set_xlabel('Forecast time')
ax[1,0].autoscale(False)

ax[1,1].imshow((mat_ricker_perfect.transpose()[0:,:] > 0.05), cmap='plasma_r')
#ax[0,0].colorbar()
ax[1,1].set_ylabel('Day of forecast initialization')
ax[1,1].set_xlabel('Forecast time')

fig.delaxes(ax[1, 2])
plt.tight_layout()

plt.savefig(f'results/{scenario}_{process}/horizon_maps.pdf')


#=====================================================#
# Forecasting with the fitted at different lead times #
#=====================================================#

data = ForecastData(y_test,x_test, climatology)
forecastloader = DataLoader(data, batch_size=1, shuffle=False, drop_last=True)

mat_ricker = np.full((len(y_test), len(y_test)), np.nan)
mat_climatology = np.full((len(y_test), len(y_test)), np.nan)

i = 0
fh_metric = 'crps'
for states, temps, clim in forecastloader:

    print('I is: ', i)
    N0 = states[:,0]
    clim = clim.squeeze().detach().numpy()
    forecast = []
    for modelfit in modelfits:
        forecast.append(modelfit.forecast(N0, temps).detach().numpy())
    forecast = np.array(forecast).squeeze()
    states = states.squeeze().detach().numpy()

    if fh_metric == 'crps':
        performance = [CRPS(forecast[:,i], states[i]).compute()[0] for i in range(forecast.shape[1])]
        performance_ref = [CRPS(clim[:, i], states[i]).compute()[0] for i in range(clim.shape[1])]
        mat_ricker[i, i:] = performance
        mat_climatology[i, i:] = performance_ref
    elif fh_metric == 'nashsutcliffe':
        performance = [np.mean([nash_sutcliffe(states[:k+1], forecast[j, :k+1]) for j in range(forecast.shape[0])]) for k in range(forecast.shape[1]-1)]
        performance_ref = [np.mean([nash_sutcliffe(states[:k+1], clim[j, :k+1]) for j in range(forecast.shape[0])])  for k in range(clim.shape[1]-1)]
        mat_ricker[i, i+1:] = performance
        mat_climatology[i, i+1:] = performance_ref
    elif fh_metric == 'mse':
        performance = mse(states[np.newaxis,:], forecast)
        performance_ref = mse(states[np.newaxis,:], clim)
        mat_ricker[i, i:] = performance
        mat_climatology[i, i:] = performance_ref
    elif fh_metric == 'correlation':
        w = 3
        performance = np.mean(rolling_corrs(states[np.newaxis,:], forecast, window=w), axis=0)
        performance_ref = np.mean(rolling_corrs(states[np.newaxis,:], clim, window=w), axis=0)
        mat_ricker[i, :i + w] = np.nan
        mat_ricker[i, i+w:] = performance
        mat_climatology[i, :i + w] = np.nan
        mat_climatology[i, i+w:] = performance_ref
    i += 1

mat_ricker[np.isinf(mat_ricker)] = np.nan
mat_ricker_plot = mat_ricker
if fh_metric == 'nashsutcliffe':
    fh = mat_ricker <= 0
    mask = np.isfinite(mat_ricker)
    mat_ricker_plot[mask] = fh[mask]
fig, ax = plt.subplots()
plt.imshow(mat_ricker_plot)
plt.colorbar()
plt.xlabel('Day of year')
plt.ylabel('Forecast length')
#plt.savefig(f'plots/horizonmap_ricker_{process}_{scenario}_{fh_metric}fh.pdf')
plt.close()

mat_climatology[np.isinf(mat_climatology)] = np.nan
mat_climatology_plot = mat_climatology
if fh_metric == 'nashsutcliffe':
    fh = mat_climatology <= 0
    mask = np.isfinite(mat_climatology)
    mat_climatology_plot[mask] = fh[mask]
fig, ax = plt.subplots()
plt.imshow(mat_climatology_plot)
plt.colorbar()
plt.xlabel('Day of year')
plt.ylabel('Forecast length')
#plt.savefig(f'plots/horizonmap_climatology_{process}_{scenario}_{fh_metric}fh.pdf')
plt.close()

if fh_metric != 'correlation':
    fig, ax = plt.subplots()
    skill = 1 - (mat_ricker/mat_climatology) # If mse of climatology is larger, term is larger zero.
    fh = skill <= 0 # FH of the ricker is reached when mse of climatology drops below ricker mse
    mask = np.isfinite(skill)
    skill[mask] = fh[mask]
    plt.imshow(skill, cmap='autumn_r')
    plt.colorbar()
    plt.xlabel('Day of year')
    plt.ylabel('Forecast length')
    #plt.savefig(f'plots/horizonmap_skill_{process}_{scenario}_{fh_metric}fh.pdf')
else:
    fig, ax = plt.subplots()
    skill = mat_ricker    # If mse of climatology is larger, term is larger zero.
    fh = skill < 0.5 # FH of the ricker is reached when mse of climatology drops below ricker mse
    mask = np.isfinite(skill)
    skill[mask] = fh[mask]
    plt.imshow(skill, cmap='autumn_r')
    plt.colorbar()
    plt.xlabel('Day of year')
    plt.ylabel('Forecast length')

plt.plot([np.argmax(skill[i,i:]) for i in range(skill.shape[0])])
