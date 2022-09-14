#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 10:22:25 2022

@author: Marieke_Wesselkamp
"""
import random
import pandas as pd

def add_history(X, Y, batchsize, history):
    """
    Creates Mini-batches from training data set.
    Used in: dev_mlp.train_model_CV
    """
    if batchsize:
        subset = [j for j in random.sample(range(X.shape[0]), batchsize) if j > history]
        subset_h = [item for sublist in [list(range(j - history, j)) for j in subset] for item in sublist]
        x = list((X[subset], X[subset_h]))  # np.concatenate((X[subset], X[subset_h]), axis=0)
        y = Y[subset]
    else:
        x = X[history:]
        y = Y[history:]
        for i in range(1, history + 1):
            outx = pd.merge(x, X.shift(periods=i)[history:], left_index=True, right_index=True)
            # outy = pd.merge(y, Y.shift(periods=i)[history:], left_index=True, right_index=True)
        x = outx

    return x, y
