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

import models 
import utils 

import os.path

#%%

def train(hparams, model_design, X, Y, data,
                   data_dir="models/mlp", splits=5):
    
    """
    
    
    """
    epochs = hparams["epochs"]
    
    kf = KFold(n_splits=splits, shuffle = False)
    kf.get_n_splits(X)
    
    #rmse_train = np.zeros((splits, epochs))
    #rmse_val = np.zeros((splits, epochs))
    mae_train = np.zeros((splits, epochs))
    mae_val = np.zeros((splits, epochs))
        
    i = 0
    
    #performance = []
    #y_tests = []
    #y_preds = []
    
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        X_test = torch.tensor(X_test).type(dtype=torch.float)
        y_test = torch.tensor(y_test).type(dtype=torch.float)
        X_train = torch.tensor(X_train).type(dtype=torch.float)
        y_train = torch.tensor(y_train).type(dtype=torch.float)
        
        model = models.MLP(model_design["layer_sizes"])
                    
        optimizer = optim.Adam(model.parameters(), lr = hparams["learningrate"])
        criterion = nn.MSELoss()
        
        #early_stopping = utils.EarlyStopping()
        
        for epoch in range(epochs):
            
            # Training
            model.train()

            x, y = utils.create_batches(X_train, y_train, hparams["batchsize"], hparams["history"])
            
            x = torch.tensor(x).type(dtype=torch.float)
            y = torch.tensor(y).type(dtype=torch.float)
            
            
            output = model(x)
            
            # Compute training loss
            loss = criterion(output, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # Evaluate current model at test set
            model.eval()
            
            with torch.no_grad():
                pred_train = model(X_train)
                pred_test = model(X_test)
                #rmse_train[i, epoch] = utils.rmse(y_train, pred_train)
                #rmse_val[i, epoch] = utils.rmse(y_test, pred_test)
                val_loss = metrics.mean_absolute_error(y_test, pred_test)  
                #early_stopping(val_loss)
                #if early_stopping.early_stop:
                #    break
                
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

    
        torch.save(model.state_dict(), os.path.join(data_dir, f"{data}_model{i}.pth"))
        
        #y_tests.append(y_test.numpy())
        #y_preds.append(preds_test.numpy())
        
    
        i += 1
    
    running_losses = {"mae_train":mae_train, "mae_val":mae_val } #, "rmse_val":rmse_val, "rmse_train":rmse_train, }

    return running_losses #, performance #, y_tests, y_preds