import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchdiffeq import odeint as odeint
import numpy as np

np.random.seed(42)

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
params = sp.symbols('pi timesteps freq')
equation1 = sp.sin(2 * params[0] * params[1] / params[3] * (t / params[1]))
sp.diff(equation1, t)

def temp(t, timesteps):
    # pi isn't built in torch.
    return torch.sin(2 * 3.14159 * timesteps / 365 * (t / timesteps))
def temp_diff(t):
    return 2*3.14159*torch.cos(2*3.14159*t/365)/365
plt.plot(temp(ts, 365))
plt.plot(temp_diff(ts))
