import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from simulations import simulate_temperature, generate_data
import numpy as np

np.random.seed(42)

#==============================================#
# Simple example for the use of the optimizer  #
#==============================================#

class Ricker_Predation(nn.Module):
    """

    """

    def __init__(self, params, noise):
        super().__init__()
        if not noise is None:
            self.model_params = torch.nn.Parameter(torch.tensor(params+noise, requires_grad=True, dtype=torch.double))
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
                                + sigma*torch.normal(mean=torch.tensor([0.0,]), std=torch.tensor([1.0]))
                out[1, i + 1] = out.clone()[1, i] * torch.exp(alpha2*(1 - beta2*out.clone()[1, i] - gamma2*out.clone()[0, i] + bx2 * Temp[i] + cx2 * Temp[i]**2)) \
                                + sigma*torch.normal(mean=torch.tensor([0.0,]), std=torch.tensor([1.0]))
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

    def __init__(self, params, noise=None):

        super().__init__()

        if not noise is None:
            self.model_params = torch.nn.Parameter(torch.tensor(params + noise, requires_grad=True, dtype=torch.double))
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

        out = torch.zeros((len(N0), len(Temp)), dtype=torch.double)
        out[:,0] = N0  # initial value

        if not self.noise is None:
            for i in range(len(Temp) - 1):
                out[:,i + 1] = out.clone()[:,i] * torch.exp(
                    alpha * (1 - beta * out.clone()[:,i] + bx * Temp[i] + cx * Temp[i] ** 2)) \
                             + sigma * torch.normal(mean=torch.tensor([0.0, ]), std=torch.tensor([1.0]))
        else:
            for i in range(len(Temp) - 1):
                out[:,i + 1] = out.clone()[:,i] * torch.exp(
                    alpha * (1 - beta * out.clone()[:,i] + bx * Temp[i] + cx * Temp[i] ** 2))

        return out

    def __repr__(self):
        return f" alpha: {self.model_params[0].item()}, \
            beta: {self.model_params[1].item()}, \
                bx: {self.model_params[2].item()}, \
                    cx: {self.model_params[3].item()}, \
               sigma: {self.noise}"



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

timesteps = 365*2
temperature = simulate_temperature(timesteps=timesteps)
observation_params = [0.18, 1, 0.021, 0.96, 1.05, 0.16,  1, 0.02, 0.97, 1.06]
noise = [0., 0]
observation_model = Ricker_Predation(params = observation_params, noise = None)
dyn_obs = observation_model(Temp = temperature)
y = dyn_obs[0,:].clone().detach().requires_grad_(True)
plt.plot(y.detach().numpy())

step_length = 150
data = SimODEData(step_length=step_length, y = y, temp=temperature)
trainloader = DataLoader(data, batch_size=1, shuffle=False, drop_last=True)

initial_params = [0.15, 0.95, 0.05, 0.05]
noise = [0.01]
model = Ricker_Ensemble(params=initial_params, noise = None)

optimizer = torch.optim.Adam([{'params':model.model_params}], lr=0.0001)
#criterion = torch.nn.MSELoss()
#criterion = CRPSLoss()

losses = []
ensemble_size = 15
for epoch in range(50):
    for batch in trainloader:

        target, temp = batch
        target = target.squeeze()
        initial_state = target.clone()[0]
        initial_ensemble = initial_state + torch.normal(torch.zeros((ensemble_size)), torch.repeat_interleave(torch.tensor([0.01]), ensemble_size))

        optimizer.zero_grad()

        output = model(initial_ensemble, temp)

        loss = torch.zeros((1), requires_grad = True).clone()
        #for i in range(step_length):
            #loss += criterion(output[:,i].squeeze(), target[i])
        loss = torch.stack([crps_loss(output[:,i].squeeze(), target[i]) for i in range(step_length)])
        loss = torch.sum(loss)/step_length
        #mse_loss = mse_criterion(output, target.squeeze())
        #mse_loss = CRPSLoss(output, target.squeeze())
        #kl_loss = kl_criterion(output, target.squeeze())
        loss.backward()
        #kl_loss.backward()
        losses.append(loss.clone())
        optimizer.step()

plt.plot(losses)
print(model)

params = [par for par in model.model_params]
modelinit = Ricker_Ensemble(params = initial_params, noise=None)
modelfit = Ricker_Ensemble(params = params, noise=None)
initial_ensemble = 1 + torch.normal(torch.zeros((ensemble_size)), torch.repeat_interleave(torch.tensor([0.1]), ensemble_size))

ypreds = modelfit.forward(N0=initial_ensemble, Temp=temperature).detach().numpy()
yinit = modelinit.forward(N0=initial_ensemble, Temp=temperature).detach().numpy()
plt.plot(np.transpose(ypreds), color='blue')
plt.plot(np.transpose(yinit), color='red')