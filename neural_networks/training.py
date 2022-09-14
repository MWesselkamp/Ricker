#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 09:11:31 2022

@author: Marieke_Wesselkamp
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import KFold

import neural_networks.models_nn as models
import neural_networks.utils as utils

import os.path

#%%

def train(hparams, model_design, X, Y, task,
                   data_dir='Users/Marieke_Wesselkamp/PycharmProjects/Ricker/models', splits=5):
    
    """
    Training loop. Unfinished for the AE!

    Missing:
    - Flexible architecture Search for encoder-decoder architecture and bottleneck size.
    (- Early stopping)
    - Dropout for Denoising AE.
    """

    epochs = hparams["epochs"]
    # Set up a blocked 5-fold cross-validation
    kf = KFold(n_splits=splits, shuffle = False)
    kf.get_n_splits(X)

    mae_train = np.zeros((splits, epochs))
    mae_val = np.zeros((splits, epochs))
        
    i = 0
    
    #performance = []
    #y_tests = []
    #y_preds = []
    
    for train_index, test_index in kf.split(X):
        
        x_train, x_test = torch.Tensor(X[train_index]), torch.Tensor(X[test_index])
        y_train, y_test = torch.Tensor(Y[train_index]), torch.Tensor(Y[test_index])

        if task == "MLP":
            model = models.MLP(model_design["layer_sizes"])
        else:
            model = models.AE(model_design["input_dimension"], model_design["output_dimension"])
                    
        optimizer = optim.Adam(model.parameters(), lr = hparams["learningrate"])
        criterion = nn.MSELoss() # AE: mean squared error as reconstruction loss?

        for epoch in range(epochs):
            
            # Training
            model.train()

            x, y = utils.create_batches(x_train, y_train, hparams["batchsize"])
            
            #x = torch.Tensor(x)#.type(dtype=torch.float)
            #y = torch.Tensor(y)#.type(dtype=torch.float)
            
            if task == "AE":
                output, latent = model(x)
            else:
                output = model(x)

            # Compute training loss
            loss = criterion(output, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # Evaluate current model at test set
            model.eval()
            
            with torch.no_grad():
                pred_train = model(x_train)
                pred_test = model(x_test)
                val_loss = metrics.mean_absolute_error(y_test, pred_test)
                
                mae_train[i, epoch] = metrics.mean_absolute_error(y_train, pred_train)
                mae_val[i, epoch] = val_loss
                
                    
         
        # Predict with fitted model
        #with torch.no_grad():
        #    preds_train = model(X_train)
        #    preds_test = model(X_test)
        #    performance.append([utils.rmse(y_train, preds_train),
        #                        utils.rmse(y_test, preds_test),
        #                        metrics.mean_absolute_error(y_train, preds_train.numpy()),
        #                        metrics.mean_absolute_error(y_test, preds_test.numpy())])

        if not data_dir is None:
            torch.save(model.state_dict(), os.path.join(data_dir, f"{task}_model{i}.pth"))
        
        #y_tests.append(y_test.numpy())
        #y_preds.append(preds_test.numpy())
        
    
        i += 1
    
    running_losses = {"mae_train":mae_train, "mae_val":mae_val } #, "rmse_val":rmse_val, "rmse_train":rmse_train, }

    return running_losses #, performance #, y_tests, y_preds