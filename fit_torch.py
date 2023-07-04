import torch
import torch.nn as nn
from torch.functional import F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from torchdiffeq import odeint as odeint
from simulations import simulate_temperature, generate_data
import numpy as np


def ricker(N0, Temp, params):

    alpha, beta, bx, cx = params
    out = []
    N = N0
    Temp = Temp.squeeze()
    for i in range(len(Temp)):
        N = N * torch.exp(alpha * (1 - beta * N + bx * Temp[i] + cx * Temp[i] ** 2))
        out.append(N)
    out = torch.tensor(out, requires_grad=True, dtype=torch.float)
    return out

class Ricker_ODE(nn.Module):
    """
     Thanks to: https://towardsdatascience.com/differential-equations-as-a-pytorch-neural-network-layer-7614ba6d587f
    """
    def __init__(self,
                 alpha: float = 0.5,  # The alpha parameter of the Lotka-Volterra system
                 beta: float = 1.0,  # The beta parameter of the Lotka-Volterra system
                 bx: float = 0.8,  # The delta parameter of the Lotka-Volterra system
                 cx: float = 0.9,  # The gamma parameter of the Lotka-Volterra system
                 timesteps: int = 150
                 ) -> None:
        super().__init__()
        self.model_params = torch.nn.Parameter(torch.tensor([alpha, beta, bx, cx, timesteps]))

    def temp(self, t, timesteps):
        # pi isn't built in torch.
        return torch.sin(2 * 3.14159 * timesteps / 365 * (t / timesteps))

    def temp_diff(self, t):
        return 100 * 2 * 3.14159 * torch.cos(2 * 3.14159 * t / 365) / 365

    def forward(self, t, state):

        alpha, beta, bx, cx, timesteps = self.model_params

        # coefficients are part of tensor model_params
        x = state[..., 0]
        sol = torch.zeros_like(state)

        sol[...,0] = x * torch.exp(alpha * (1 - beta * x + bx * self.temp_diff(t) + cx * self.temp_diff(t) ** 2))

        return sol

    def __repr__(self):
        return f" alpha: {self.model_params[0].item()}, \
            beta: {self.model_params[1].item()}, \
                delta: {self.model_params[2].item()}, \
                    gamma: {self.model_params[3].item()}"


timesteps = 365
ricker_model = Ricker_ODE(alpha=3.7, timesteps=timesteps) # use default parameters
ts = torch.linspace(0,float(timesteps),timesteps)
batch_size = 5
initial_conditions = torch.tensor([1]) + 0.50*torch.randn((batch_size, 1))
solr = odeint(ricker_model, initial_conditions, ts, method = 'dopri5').detach().numpy()

plt.plot(ts, solr.squeeze(), lw=0.5)
plt.title("Time series of the Ricker equation")
plt.xlabel("time")
plt.ylabel("x")


import sympy as sp

t = sp.symbols('t')
params = sp.symbols('pi timesteps')
equation1 = sp.sin(2 * params[0] * params[1] / 365 * (t / params[1]))
sp.diff(equation1, t)

def temp(t, timesteps):
    # pi isn't built in torch.
    return torch.sin(2 * 3.14159 * timesteps / 365 * (t / timesteps))
def temp_diff(t):
    return 2*3.14159*torch.cos(2*3.14159*t/365)/365
plt.plot(temp(ts, 365))
plt.plot(temp_diff(ts))

#==============================================#
# Simple example for the use of the optimizer  #
#==============================================#

class Ricker_Predation(nn.Module):
    """
     The Lotka-Volterra equations are a pair of first-order, non-linear, differential equations
     describing the dynamics of two species interacting in a predator-prey relationship.
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

def my_custom_loss(outputs, labels):

    loss = torch.mean(torch.abs(outputs - labels))
    return loss

timesteps = 365
temperature = simulate_temperature(timesteps=timesteps)
observation_params = [0.18, 1, 0.021, 0.96, 1.05, 0.16,  1, 0.02, 0.97, 1.06]
noise = [0., 0]
observation_model = Ricker_Predation(params = observation_params, noise = None)
dyn_obs = observation_model(Temp = temperature)
y = dyn_obs[0,:].clone().detach().requires_grad_(True)
plt.plot(y.detach().numpy())

data = SimODEData(step_length=10, y = y, temp=temperature)
trainloader = DataLoader(data, batch_size=1, shuffle=True, drop_last=True)

initial_params = [0.15, 0.95, 0.05, 0.05]
noise = [0.01]
model = Ricker(params=initial_params, noise = None)

optimizer = torch.optim.Adam([{'params':model.model_params}], lr=0.001)
mse_criterion = torch.nn.MSELoss()
kl_criterion = torch.nn.KLDivLoss(reduction='batchmean')

losses = []
for epoch in range(5):
    for batch in trainloader:
        target, temp = batch
        initial_state = target.clone()[:,0]
        optimizer.zero_grad()
        output = model(initial_state, temp)
        #mse_loss = mse_criterion(output, target.squeeze())
        mse_loss = my_custom_loss(output, target.squeeze())
        #kl_loss = kl_criterion(output, target.squeeze())
        mse_loss.backward()
        #kl_loss.backward()
        losses.append(mse_loss.clone())
        optimizer.step()

plt.plot(losses)
print(model)


#===============#
# Fit with CPRS #
#===============#

xsim, xobs = generate_data(timesteps=150, growth_rate=0.1,
                                           sigma=0.01, phi=0.00, initial_uncertainty=0.001,
                                           doy_0=0, ensemble_size=15)

xsim = torch.tensor(xsim)
xobs = torch.tensor(xobs[:,:,0])

crps = CRPS(xsim[:,0], xobs[:,0])
crps.compute()