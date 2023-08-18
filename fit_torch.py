import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import argparse
import os
import os.path
import yaml
import scipy.stats as stats

from utils import simulate_temperature, standardize
from visualisations import plot_fit
from torch.utils.data import DataLoader, Dataset
from CRPS import CRPS
import properscoring as ps
from metrics import rolling_mse, mse, rolling_corrs, rmse, absolute_differences
from sklearn.metrics import r2_score
from scipy.stats import shapiro
from neuralforecast.losses.pytorch import sCRPS, MQLoss, MAE, QuantileLoss

#parse = False
seed = False

if seed:
    np.random.seed(42)
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


class SimODEData(Dataset):
    """
        A very simple dataset class for simulating ODEs
    """

    def __init__(self,
                 step_length,  # List of time points as tensors
                 y,  # List of dynamical state values (tensor) at each time point
                 y_sigma,
                 temp,
                 ):
        self.step_length = step_length
        self.y = y
        self.y_sigma = y_sigma
        self.temp = temp

    def __len__(self) -> int:
        return len(self.y) - self.step_length

    def __getitem__(self, index: int): #  -> Tuple[torch.Tensor, torch.Tensor]
        return self.y[index:index+self.step_length], self.y_sigma[index:index+self.step_length], self.temp[index:index+self.step_length]
class ForecastData(Dataset):
    """
        A very simple dataset class for generating forecast data sets of different lengths.
    """
    def __init__(self, y, temp, climatology = None, forecast_days = 'all', lead_time = None):
        self.y = y
        self.temp = temp
        self.climatology = climatology
        self.forecast_days = forecast_days
        self.lead_time = lead_time

    def __len__(self) -> int:
        if self.forecast_days == 'all':
            return len(self.y)-1
        else:
            return self.forecast_days

    def __getitem__(self, index: int): #  -> Tuple[torch.Tensor, torch.Tensor]
        if self.forecast_days == 'all':
            if not self.climatology is None:
                return self.y[index:len(self.y)], self.temp[index:len(self.temp)], self.climatology[:,index:len(self.y)]
            else:
                return self.y[index:len(self.y)], self.temp[index:len(self.temp)]
        else:
            if not self.climatology is None:
                return self.y[index:(index+self.lead_time)], self.temp[index:(index+self.lead_time)], self.climatology[:,index:(index+self.lead_time)]
            else:
                return self.y[index:(index+self.lead_time)], self.temp[index:(index+self.lead_time)]

def nash_sutcliffe(observed, modeled):
    """
    Calculate Nash-Sutcliffe Efficiency (NSE).
    """
    observed = np.array(observed)
    modeled = np.array(modeled)
    mean_observed = np.mean(observed)

    # Calculate sum of squared differences between observed and modeled values
    ss_diff = np.sum((observed - modeled) ** 2)
    # Calculate sum of squared differences between observed and mean of observed values
    ss_total = np.sum((observed - mean_observed) ** 2)

    # Nash-Sutcliffe Efficiency
    nse = 1 - (ss_diff / ss_total)

    return nse

def rolling_nash_sutcliffe(reference, ensemble, window=2):
    nse = [[nash_sutcliffe(reference[j:j + window], ensemble[i, j:j + window]) for i in range(ensemble.shape[0])] for
             j in range(ensemble.shape[1] - window)]
    nse = np.transpose(np.array(nse))
    return nse
class LogNormalLoss(nn.Module):
    def __init__(self, sigma_true):
        super(LogNormalLoss, self).__init__()
        self.sigma_true = sigma_true

    def forward(self, predictions, mu_true):
        # Calculate the log-likelihood for a log-normal distribution
        loss = 0.5 * ((torch.log(predictions) - mu_true) ** 2) / self.sigma_true**2
        loss = torch.mean(loss)

        return loss

def crps_loss(outputs, targets):

    fc = torch.sort(outputs).values
    ob = targets.clone()
    m = len(fc)

    cdf_fc = torch.zeros_like(fc)
    cdf_ob = torch.zeros_like(fc)
    delta_fc = torch.zeros_like(fc)
    # do for all ensemble members
    for f in range(len(fc) - 1):
        # check is ensemble member and its following ensemble member is smaller than observation.
        if (fc[f] < ob) and (fc[f + 1] < ob):
            cdf_fc[f] = ((f + 1) * 1 / m)
            cdf_ob[f] = 0
            delta_fc[f] = (fc[f + 1] - fc[f])
        elif (fc[f] < ob) and (fc[f + 1] > ob):
            # check is ensemble member is smaller than observation and its following ensemble member is larger than observation.
            cdf_fc[f] = ((f + 1) * 1 / m)
            cdf_fc[f] = ((f + 1) * 1 / m)
            cdf_ob[f] = 0
            cdf_ob[f] = 1
            delta_fc[f] = ob - fc[f]
            delta_fc[f] = fc[f + 1] - ob
        else:
            cdf_fc[f] = ((f + 1) * 1 / m)
            cdf_ob[f] = 1
            delta_fc[f] =  fc[f + 1] - fc[f]

    loss = torch.sum(((cdf_fc - cdf_ob) ** 2) * delta_fc)

    return loss

def train(y_train,sigma_train, x_train, model, epochs, loss_fun = 'mse', step_length = 2):

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

            if (loss_fun == 'mse') & (sigma_train is not None):
                loss = criterion(output, target) + criterion(output_sigma[0], target_upper) + criterion(output_sigma[1], target_lower)
            elif (loss_fun == 'mse') & (sigma_train is not None):
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

def plot_losses(losses, loss_fun, log=True, saveto=''):
    if log:
        ll = np.log(torch.stack(losses).detach().numpy())
    else:
        ll = torch.stack(losses).detach().numpy()
    plt.plot(ll)
    plt.ylabel(f'{loss_fun} loss')
    plt.xlabel(f'Epoch')
    plt.savefig(os.path.join(saveto, 'losses.pdf'))

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

def plot_posterior(df, saveto=''):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    axes[0, 0].hist(df["alpha"], bins=10, edgecolor='black')
    axes[0, 0].set_title("Histogram of alpha")
    axes[0, 1].hist(df["beta"], bins=10, edgecolor='black')
    axes[0, 1].set_title("Histogram of beta")
    axes[1, 0].hist(df["bx"], bins=10, edgecolor='black')
    axes[1, 0].set_title("Histogram of bx")
    axes[1, 1].hist(df["cx"], bins=10, edgecolor='black')
    axes[1, 1].set_title("Histogram of cx")
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(saveto, 'posterior.pdf'))

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
        losses = train(y_train, sigma_train, x_train, model, epochs=epochs, loss_fun = loss_fun, step_length = step_length)
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
    return yinit, ypreds, modelfits

save = True

observation_params, initial_params, true_noise, initial_noise = set_parameters(process = 'stochastic', scenario = 'chaotic')
y_train, y_test, sigma_train, sigma_test, x_train, x_test, climatology = create_observations(years = 10, observation_params= observation_params, true_noise = true_noise)
fitted_values, losses = fit_models(y_train, x_train, sigma_train, initial_params, initial_noise, samples=10, epochs=20, loss_fun = 'mse', step_length = 10)
posterior = pd.DataFrame(fitted_values)

fitted_pars = posterior.mean().values
ip = np.array([np.random.normal(i, 0.01, 20) for i in fitted_pars])
keys = ['alpha', 'beta', 'bx', 'cx', 'sigma', 'phi']
ip_samples = [dict(zip(keys, ip[:,column])) for column in range(ip.shape[1])]

if save:
    pd.DataFrame(fitted_values).to_csv('results/chaotic_stochastic/fitted_values.csv')
    plot_posterior(posterior, saveto='results/chaotic_stochastic_mse')
    plot_losses(losses, loss_fun='mse', saveto='results/chaotic_stochastic/fitted_values.csv')

yinit, ypreds, modelfits = forecast_fitted(y_test, x_test, ip_samples, initial_params, initial_noise, initial_uncertainty = 0.001)
ypreds = ypreds[~np.any(np.isnan(ypreds), axis=1),:]

pointwise_mse = {'MSE': mse(y_test.detach().numpy()[np.newaxis,:], ypreds),
                  'MSE_clim': mse(y_test.detach().numpy()[np.newaxis,:], climatology.detach().numpy())}
pointwise_crps = {'CRPS': [CRPS(ypreds[:,i], y_test.detach().numpy()[i]).compute()[0] for i in range(ypreds.shape[1])],
                  'CRPS_clim': [CRPS(climatology.detach().numpy()[:,i], y_test.detach().numpy()[i]).compute()[0] for i in range(ypreds.shape[1])]}
plot_fit(yinit, ypreds[:40,:], y_test, scenario=f"{'chaotic'}_{'stochastic'}", loss_fun='mse', clim=climatology,fh_metric1=pointwise_mse,fh_metric2=pointwise_crps, save=True)

sr = [shapiro(ypreds[:,i])[1] for i in range(ypreds.shape[1])]
plt.plot(sr)


fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(6, 8), sharex=True)
for i in range(7):
    x = ypreds[:,i]
    mu = x.mean()
    sigma = x.std()
    axes[i].hist(x, alpha=0.5)
    xs = np.linspace(0.93, 1.05, num=100)
    axes[i].plot(xs, stats.norm.pdf(xs, mu, sigma))
    axes[i].vlines(x = y_test[i].detach().numpy(), ymin = 0, ymax = 50)
    axes[i].vlines(x = climatology[:,i].detach().numpy(), ymin = 0, ymax = 50, colors='lightblue')

for i in range(5):
    print('Ensemble', ps.crps_ensemble(y_test[i].detach().numpy(), ypreds[:,i]))
    print('Climatology', ps.crps_ensemble(y_test[i].detach().numpy(), climatology.detach().numpy()[:,i]))

#=============================#
# Forecasting with the fitted #
#=============================#
def get_fh(fh_metric, ypreds, y_test, climatology, skillscore = True, fh_def = 'actual'):

    forecast = ypreds
    states = y_test.detach().numpy()
    clim = climatology.detach().numpy()
    perfect_observation = np.mean(forecast, axis=0)

    if fh_metric == 'crps':
        performance = [CRPS(forecast[:,i], states[i]).compute()[0] for i in range(forecast.shape[1])]
        performance_perfect = [CRPS(forecast[:,i], perfect_observation[i]).compute()[0] for i in range(forecast.shape[1])]
        performance_ref = [CRPS(clim[:, i], states[i]).compute()[0] for i in range(clim.shape[1])]
        performance_ref_perfect = [CRPS(clim[:, i], perfect_observation[i]).compute()[0] for i in range(clim.shape[1])]

    elif fh_metric == 'nashsutcliffe':
        performance = [nash_sutcliffe(states[np.newaxis,:], forecast[:,i]) for i in range(forecast.shape[1])]
        performance_ref = [np.mean([nash_sutcliffe(states[:k+1], clim[j, :k+1]) for j in range(forecast.shape[0])])  for k in range(clim.shape[1]-1)]

    elif fh_metric == 'abs':
        performance = np.mean(absolute_differences(states[np.newaxis,:], forecast), axis=0)
        performance_perfect = np.mean(absolute_differences(perfect_observation, forecast), axis=0)
        performance_ref =  np.mean(absolute_differences(states[np.newaxis,:], clim), axis=0)
        performance_ref_perfect =  np.mean(absolute_differences(perfect_observation, clim), axis=0)

    elif fh_metric == 'mse':
        performance = mse(states[np.newaxis,:], forecast)
        performance_perfect = mse(perfect_observation, forecast)
        performance_ref = mse(states[np.newaxis,:], clim)
        performance_ref_perfect = mse(perfect_observation, clim)

    elif fh_metric == 'rmse':
        performance = rmse(states[np.newaxis,:], forecast)
        performance_perfect = rmse(perfect_observation, forecast)
        performance_ref = rmse(states[np.newaxis,:], clim)
        performance_ref_perfect = rmse(perfect_observation, clim)

    elif fh_metric == 'rsquared':
        performance = [[r2_score(states[:j], forecast[i,:j]) for i in range(forecast.shape[0])] for j in range(1,forecast.shape[1])]
        performance_ref = [[r2_score(states[:j], clim[i,:j]) for i in range(clim.shape[0])] for j in range(1,clim.shape[1])]

    elif fh_metric == 'corr':
        w = 3
        performance = np.mean(rolling_corrs(states[np.newaxis,:], forecast, window=w), axis=0)
        performance_perfect = np.mean(rolling_corrs(perfect_observation[np.newaxis,:], forecast, window=w), axis=0)
        performance_ref = np.mean(rolling_corrs(states[np.newaxis,:], clim, window=w), axis=0)
        performance_ref_perfect = np.mean(rolling_corrs(perfect_observation[np.newaxis,:], clim, window=w), axis=0)

    performance = np.array(performance)
    performance_perfect = np.array(performance_perfect)
    performance_ref = np.array(performance_ref)
    performance_ref_perfect = np.array(performance_ref_perfect)
    if skillscore:
        if fh_def == 'actual':
            skill = 1 - (performance/performance_ref) # If mse of climatology is larger, term is larger zero.
        else:
            skill = 1 - (performance_perfect/performance_ref_perfect)
        plt.plot(skill)
        if fh_metric == "crps":
            fh = np.argmax(skill < 0)
        elif fh_metric == 'abs':
            fh = np.argmax(skill < 0)
        elif fh_metric == 'mse':
            fh = np.argmax(skill < 0)
        elif fh_metric == 'rmse':
            fh = np.argmax(skill < 0)
        elif fh_metric == 'nashsutcliffe':
            fh = np.argmax(skill < 0)
        elif fh_metric == 'corr':
            if fh_def == 'actual':
                fh = np.argmax(performance < 0.5)
            else:
                fh = np.argmax(performance_perfect < 0.5)

        return fh

    else:
        if fh_def == 'actual':
            proficiency = performance
        else:
            proficiency = performance_perfect

        if fh_metric == "crps":
            fh = np.argmax(proficiency > 0.05)
        elif fh_metric == 'abs':
            fh = np.argmax(proficiency > 0.1)
        elif fh_metric == 'mse':
            fh = np.argmax(proficiency > 0.1)
        elif fh_metric == 'rmse':
            fh = np.argmax(proficiency > 0.1)
        elif fh_metric == 'nashsutcliffe':
            fh = np.argmax(proficiency > 0.1)
        elif fh_metric == 'corr':
            fh = np.argmax(proficiency < 0.5)

        return fh

metrics = ['corr', 'mse', 'abs', 'crps']
fh_a = [get_fh(m, ypreds, y_test, climatology, skillscore=False,fh_def='actual') for m in metrics]
fh_p = [get_fh(m, ypreds, y_test, climatology, skillscore=False,fh_def='perfect') for m in metrics]
fhs_a = [get_fh(m, ypreds, y_test, climatology, skillscore=True, fh_def='actual') for m in metrics]
fhs_p = [get_fh(m, ypreds, y_test,climatology, skillscore=True,fh_def='perfect') for m in metrics]
pd.DataFrame([fh_a, fh_p, fhs_a, fhs_p], columns=metrics, index = ['fh_a', 'fh_p', 'fhs_a', 'fhs_p']).to_csv('results/chaotic_stochastic/horizons.csv')


plt.scatter(metrics, fh_p, marker='D', color='red', label='$\hat{h}$')
plt.scatter(metrics, fh_a, marker='D', color='blue', label='$h$')
plt.scatter(metrics, fhs_p, marker='D', facecolors='none', edgecolors='red', label='$\hat{h}_{s}$')
plt.scatter(metrics, fhs_a, marker='D', facecolors='none', edgecolors='blue', label='$h_{s}$')
plt.ylabel('Forecast horizon')
plt.xlabel('Metric')
plt.legend()
plt.savefig('results/chaotic_stochastic/horizons.pdf')

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

fig, ax = plt.subplots(2, 3, figsize=(14, 7), sharey=False, sharex=False)

heatmap0 = ax[0,0].imshow(mat_ricker.transpose()[0:,:])
cb = plt.colorbar(heatmap0, ax=ax[0,0])
cb.set_label('CRPS')
ax[0,0].set_ylabel('Day of forecast initialization')
ax[0,0].set_xlabel('Time horizon/Lead time')

ax[0,1].imshow((mat_ricker.transpose()[0:,:] > 0.05))
#plt.colorbar()
ax[0,1].set_ylabel('Day of forecast initialization')
ax[0,1].set_xlabel('Time horizon/Lead time')

ax[0,2].imshow((mat_ricker.transpose()[0:,:] > mat_climatology.transpose()[0:,:]))
#ax[0,2].colorbar()
ax[0,2].set_ylabel('Day of forecast initialization')
ax[0,2].set_xlabel('Time horizon/Lead time')

heatmap1 = ax[1,0].imshow(mat_ricker_perfect.transpose()[0:,:])
cb1 = plt.colorbar(heatmap1, ax=ax[1,0])
cb1.set_label('CRPS')
ax[1,0].set_ylabel('Day of forecast initialization')
ax[1,0].set_xlabel('Time horizon/Lead time')

ax[1,1].imshow((mat_ricker_perfect.transpose()[0:,:] > 0.05))
#ax[0,0].colorbar()
ax[1,1].set_ylabel('Day of forecast initialization')
ax[1,1].set_xlabel('Time horizon/Lead time')

fig.delaxes(ax[1, 2])

plt.savefig('results/chaotic_stochastic/horizon_maps.pdf')


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
