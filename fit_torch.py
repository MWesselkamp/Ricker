import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import os.path
import yaml

from utils import simulate_temperature
from visualisations import plot_fit
from torch.utils.data import DataLoader, Dataset
from CRPS import CRPS
from metrics import rolling_mse, mse
from neuralforecast.losses.pytorch import sCRPS, MQLoss, MAE, QuantileLoss

np.random.seed(42)
parse = False

#set flags
if parse:

    parser = argparse.ArgumentParser()
    parser.add_argument("--process", type=str, help="Set process type to stochastic or deterministic")
    parser.add_argument("--scenario", type=str, help="Set scenario type to chaotic or nonchaotic")
    parser.add_argument("--loss_fun", type=str, help="Set loss function to quantile, crps or mse")
    parser.add_argument("--fh_metric", type=str, help="Set horizon metric to crps, mse or nashsutcliffe")

    # Parse the command-line arguments
    args = parser.parse_args()

    process = args.process
    scenario = args.scenario
    loss_fun = args.loss_fun
    fh_metric = args.fh_metric
else:
    process = 'deterministic'
    scenario = 'chaotic'
    loss_fun = 'mse'
    fh_metric = 'mse'

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
     The Lotka-Volterra equations are a pair of first-order, non-linear, differential equations
     describing the dynamics of two species interacting in a predator-prey relationship.
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
     The Lotka-Volterra equations are a pair of first-order, non-linear, differential equations
     describing the dynamics of two species interacting in a predator-prey relationship.
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
            initial = N0 + phi*torch.normal(torch.zeros((ensemble_size)), torch.repeat_interleave(torch.tensor([0.1]), ensemble_size))
            out = torch.zeros((len(initial), len(Temp)), dtype=torch.double)
        else:
            initial = N0
            out = torch.zeros((1, len(Temp)), dtype=torch.double)

        out[:,0] = initial  # initial value

        if not self.noise is None:
            for i in range(len(Temp) - 1):
                out[:,i + 1] = out.clone()[:,i] * torch.exp(
                    alpha * (1 - beta * out.clone()[:,i] + bx * Temp[i] + cx * Temp[i] ** 2)) \
                             + sigma * torch.normal(mean=torch.tensor([0.0, ]), std=torch.tensor([0.1]))
        else:
            for i in range(len(Temp) - 1):
                out[:,i + 1] = out.clone()[:,i] * torch.exp(
                    alpha * (1 - beta * out.clone()[:,i] + bx * Temp[i] + cx * Temp[i] ** 2))

        return out

    def get_fit(self):

        return {"alpha": self.model_params[0].item(), \
            "beta": self.model_params[1].item(), \
                "bx": self.model_params[2].item(), \
                    "cx": self.model_params[3].item(), \
               "sigma": self.noise if self.noise is None else self.model_params[4].item(), \
               "phi": self.initial_uncertainty if self.initial_uncertainty is None else self.model_params[5].item()}



class SimODEData(Dataset):
    """
        A very simple dataset class for simulating ODEs
    """

    def __init__(self,
                 step_length,  # List of time points as tensors
                 y,  # List of dynamical state values (tensor) at each time point
                 temp,
                 ):
        self.step_length = step_length
        self.y = y
        self.temp = temp

    def __len__(self) -> int:
        return len(self.y) - self.step_length

    def __getitem__(self, index: int): #  -> Tuple[torch.Tensor, torch.Tensor]
        return self.y[index:index+self.step_length], self.temp[index:index+self.step_length]

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

# Create observations
timesteps = 365*2
temperature = simulate_temperature(timesteps=timesteps)

if scenario == 'chaotic':
    observation_params = [1.08, 1, 0.021, 0.41, 0.5, 1.06,  1, 0.02, 0.62, 0.72] # for chaotic, set r1 = 0.18 and r2 = 0.16 to r1 = 1.08 and r2 = 1.06
elif scenario == 'nonchaotic':
    observation_params = [0.18, 1, 0.021, 0.41, 0.5, 0.16, 1, 0.02, 0.62, 0.72]
if process == 'stochastic':
    true_noise =  0.3 # None # [1]
elif process == 'deterministic':
    true_noise = None

observation_model = Ricker_Predation(params = observation_params, noise = true_noise)
dyn_obs = observation_model(Temp = temperature)
y = dyn_obs[0,:].clone().detach().requires_grad_(True)

plt.plot(y.detach().numpy())
plt.show()
plt.close()

step_length = 25

data = SimODEData(step_length=step_length, y = y, temp=temperature)
trainloader = DataLoader(data, batch_size=1, shuffle=False, drop_last=True)

if scenario == 'chaotic':
    initial_params = [0.95, 0.95, 0.95, 0.05]
elif scenario == 'nonchaotic':
    initial_params = [0.15, 0.95, 0.15, 0.05]

if process == 'stochastic':
    initial_noise = 0.1  # None # [0.01]
elif process == 'deterministic':
    initial_noise = None # [0.01]

initial_uncertainty = 0.1
model = Ricker_Ensemble(params=initial_params, noise = initial_noise, initial_uncertainty = initial_uncertainty)

optimizer = torch.optim.Adam([{'params':model.model_params}], lr=0.0001)

criterion = torch.nn.MSELoss()
criterion2 = sCRPS()
criterion3 = MQLoss(quantiles = [0.4, 0.6])
criterion4 = QuantileLoss(0.5) # Pinball Loss

losses = []
for epoch in range(25):
    for batch in trainloader:

        target, temp = batch
        target = target.squeeze()
        initial_state = target.clone()[0]

        optimizer.zero_grad()

        output = model(initial_state, temp)

        if loss_fun == 'mse':
            loss = criterion(output, target)
            loss = torch.sum(loss) / step_length
        elif loss_fun == 'crps':
            # loss = torch.zeros((1), requires_grad=True).clone()
            loss = torch.stack([crps_loss(output[:,i].squeeze(), target[i]) for i in range(step_length)])
            loss = torch.sum(loss)/step_length
        elif loss_fun == 'quantile':
            loss = torch.stack([criterion4(target[i], output[:,i].squeeze()) for i in range(step_length)])
            loss = torch.sum(loss) / step_length
        elif loss_fun == 'mquantile':
            pass

        loss.backward()
        losses.append(loss.clone())
        optimizer.step()

def plot_losses(losses, saveto=''):

    ll = torch.stack(losses).detach().numpy()
    plt.plot(ll)
    plt.ylabel(f'{loss_fun} loss')
    plt.xlabel(f'Epoch')
    plt.savefig(os.path.join(saveto, 'losses.pdf'))
    plt.close()

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


print(model.get_fit())
fm = model.get_fit()
save_fit(fm, filename='params', losses=losses, directory_path=f'results/fit/{scenario}_{process}_{loss_fun}')

params = [v for v in fm.values()][:4]
if (not initial_noise is None) & (not initial_uncertainty is None):
    modelinit = Ricker_Ensemble(params = initial_params, noise=initial_noise, initial_uncertainty=initial_uncertainty)
    modelfit = Ricker_Ensemble(params = params, noise=fm['sigma'], initial_uncertainty=fm['phi'])
elif (not initial_noise is None) & (initial_uncertainty is None):
    modelinit = Ricker_Ensemble(params=initial_params, noise=initial_noise, initial_uncertainty=initial_uncertainty)
    modelfit = Ricker_Ensemble(params=params, noise=fm['sigma'], initial_uncertainty=None)
elif (initial_noise is None) & (not initial_uncertainty is None):
    modelinit = Ricker_Ensemble(params=initial_params, noise=initial_noise, initial_uncertainty=initial_uncertainty)
    modelfit = Ricker_Ensemble(params=params, noise=None, initial_uncertainty=fm['phi'])
else:
    modelinit = Ricker_Ensemble(params=initial_params, noise=initial_noise, initial_uncertainty=initial_uncertainty)
    modelfit = Ricker_Ensemble(params=params, noise=None, initial_uncertainty=None)

yinit = modelinit.forward(N0=1, Temp=temperature).detach().numpy()
ypreds = modelfit.forward(N0=1, Temp=temperature).detach().numpy()

plot_fit(yinit, ypreds, y, scenario=f'{process}_{scenario}', loss_fun=loss_fun)

#=============================#
# Forecasting with the fitted #
#=============================#

class ForecastData(Dataset):
    """
        A very simple dataset class for generating forecast data sets of different lengths.
    """
    def __init__(self, y, temp, climatology = None):
        self.y = y
        self.temp = temp
        self.climatology = climatology

    def __len__(self) -> int:
        return len(self.y)-1

    def __getitem__(self, index: int): #  -> Tuple[torch.Tensor, torch.Tensor]
        if not self.climatology is None:
            return self.y[index:len(self.y)], self.temp[index:len(self.temp)], self.climatology[:,index:len(self.y)]
        else:
            return self.y[index:len(self.y)], self.temp[index:len(self.temp)]
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

# Create climatology
years = 15
timesteps_clim = 365*years
temperature_clim = simulate_temperature(timesteps=timesteps_clim)
clim_model = Ricker_Predation(params = observation_params, noise = true_noise)
clim_obs = clim_model(Temp = temperature_clim)
clim_mat = clim_obs[0,:].view(years, 365)

y_test, temp_test = y[365:], temperature[365:]
temporal_error = {'MSE': mse(y_test.detach().numpy()[np.newaxis,:], ypreds[:,365:]),
                  'MSE_clim': mse(y_test.detach().numpy()[np.newaxis,:], clim_mat.detach().numpy())}
plot_fit(yinit[:,365:], ypreds[:,365:], y_test, scenario=f'{process}_{scenario}', loss_fun=loss_fun, clim=clim_mat,fh_metric=temporal_error, save=True)

data = ForecastData(y_test,temp_test, clim_mat)
forecastloader = DataLoader(data, batch_size=1, shuffle=False, drop_last=True)

mat_ricker = np.full((len(y_test), len(y_test)), np.nan)
mat_climatology = np.full((len(y_test), len(y_test)), np.nan)

i = 0
for states, temps, clim in forecastloader:
    print('I is: ', i)
    N0 = states[:,0]
    clim = clim.squeeze().detach().numpy()
    forecast = modelfit.forward(N0, temps).detach().numpy()
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
plt.savefig(f'plots/horizonmap_ricker_{process}_{scenario}_{fh_metric}fh.pdf')
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
plt.savefig(f'plots/horizonmap_climatology_{process}_{scenario}_{fh_metric}fh.pdf')
plt.close()

fig, ax = plt.subplots()
skill = 1 - (mat_ricker/mat_climatology) # If mse of climatology is larger, term is larger zero.
fh = skill <= 0 # FH of the ricker is reached when mse of climatology drops below ricker mse
mask = np.isfinite(skill)
skill[mask] = fh[mask]
plt.imshow(skill, cmap='autumn_r')
plt.colorbar()
plt.xlabel('Day of year')
plt.ylabel('Forecast length')
plt.savefig(f'plots/horizonmap_skill_{process}_{scenario}_{fh_metric}fh.pdf')
