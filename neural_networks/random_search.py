#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 09:06:42 2022

@author: Marieke_Wesselkamp
"""

import random 
import training
import numpy as np
import pandas as pd

#%%

def architecture_searchspace(input_size, output_size, gridsize,
                max_layers = 3):
    grid = []
    for i in range(gridsize):
        layersizes = [input_size]
        nlayers = random.randint(1,max_layers)
        for i in range(nlayers):
            size = random.choice([4,8,16,32])
            layersizes.append(size)
        layersizes.append(output_size)
        if layersizes not in grid:
            grid.append(layersizes)
    return grid

#%%
def architecture_search():

    grid = architecture_searchspace(7,1,40)

    hparams = {"epochs":300,
           "batchsize":16,
           "learningrate":0.01,
           "history":1}

    mae_train = []
    mae_val = []

    for i in range(len(grid)):
        model_design = {"layer_sizes":grid[i]}
        running_losses = training.train(hparams, model_design, X_P1.to_numpy(), Y_P1.to_numpy(), "randomsearch")  
        mae_train.append(np.mean(np.transpose(running_losses["mae_train"])[-1]))
        mae_val.append(np.mean(np.transpose(running_losses["mae_val"])[-1]))
        print(f"fitted model {i}")
    
    df = pd.DataFrame(grid)
    df["mae_train"] = mae_train
    df["mae_val"] = mae_val
    print("Random architecture search best result:")
    print(df.loc[[df["mae_val"].idxmin()]])
    layersizes = grid[df["mae_val"].idxmin()]
    
    return layersizes

#%%
def hparams_searchspace(gridsize):
    
    grid = []
    for i in range(gridsize):
        learning_rate = random.choice([0.01,0.05,0.001,0.005])
        batchsize = random.choice([8, 16, 32, 64])
        if [learning_rate, batchsize] not in grid:
            grid.append([learning_rate, batchsize])
    return grid

#%%
def hparams_search(layersizes):

    grid = hparams_searchspace(20)

    model_design = {"layer_sizes":layersizes}
    mae_train = []
    mae_val = []

    for i in range(len(grid)):
        
        hparams = {"epochs":300,
                   "batchsize":grid[i][1],
                   "learningrate":grid[i][0],
                   "history":1}

        running_losses = training.train(hparams, model_design, X_P1.to_numpy(), Y_P1.to_numpy(), "randomsearch")  
        mae_train.append(np.mean(np.transpose(running_losses["mae_train"])[-1]))
        mae_val.append(np.mean(np.transpose(running_losses["mae_val"])[-1]))
        print(f"fitted model {i}")
    
    df = pd.DataFrame(grid)
    df["mae_train"] = mae_train
    df["mae_val"] = mae_val
    print("Random hparams search best result:")
    print(df.loc[[df["mae_val"].idxmin()]])
    hparams = grid[df["mae_val"].idxmin()]
    
    return 