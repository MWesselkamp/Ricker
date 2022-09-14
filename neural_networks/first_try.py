# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% Packges
import os
os.chdir("/Users/Marieke_Wesselkamp/PycharmProjects/Ricker/neural_networks")
import utils
import matplotlib.pyplot as plt
import preprocessing
import torch
import numpy as np
import models

from torch.utils.data import Dataset
#%% Data
dfs = preprocessing.get_gpp()
# Prep data for linear AE
x = torch.tensor(dfs['GPP_ref'].values)
xs = x.reshape(-1,1)
input_dim = xs.shape[1]
output_dim = xs.shape[1]

#%%
history = 1
# Model Initialization
model = models.AE(history+1, output_dim)
# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()
# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-1,
                             weight_decay = 1e-8)

# %%
epochs = 50
outputs = []
losses = []
latents = []
for epoch in range(epochs):
       
    # Sample a random batch from time series with history.
    sample_x, sample_y = utils.add_history(xs, xs, batchsize=16, history=history)
    sample_x = torch.stack(sample_x, axis=1).squeeze(2)
    sample_x, sample_y = sample_x.float(), sample_y.float()
    
    # Output of Autoencoder
    latent, reconstructed = model(sample_x)
    latents.append(latent)
     
    # Calculating the loss function
    loss = loss_function(reconstructed, sample_y)
     
    # The gradients are set to zero,
    # the gradient is computed and stored.
    # .step() performs parameter update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
      
    # Storing the losses in a list for plotting
    losses.append(loss)
    outputs.append((epochs, sample_y, reconstructed))
 
# Defining the Plot Style
plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')
 
# Plotting the last 100 values
plt.plot(losses[-100:])
#%% Predict with model
x_h, y_h = utils.add_history(dfs['GPP_ref'], dfs['GPP_ref'], batchsize=None, history=1)
latent, reconstructed = model(torch.tensor(x_h.values).float())

plt.plot(np.arange(len(y_h)), y_h, label="true")
plt.plot(np.arange(len(y_h)), reconstructed.detach().numpy(), label="reconstructed")
plt.legend()


#%% Inspect model fit
# latent 
latent_ins = latent.detach().numpy()
plt.plot(np.transpose(latent_ins))