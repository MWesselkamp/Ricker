#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 10:22:25 2022

@author: Marieke_Wesselkamp
"""
import numpy as np
import random

def split(x, n = 5):
    out = np.array_split(x, n)
    x_nas = out[0]
    x_train = np.concatenate(out[1:(len(out)-2)])
    x_test = out[len(out)-1]
    return x_nas, x_train, x_test

def make_history(x, history):

    x = x[history:]
    y = x[history:]

    xh = []
    for i in range(history):
        xh.append(x[i:-(history-i)])

    return np.array(xh), y

def create_batches(x, y, batchsize):

    x = x[random.sample(range(x.shape[0]), batchsize),:]
    y = y[random.sample(range(x.shape[0]), batchsize),:]

    return x, y