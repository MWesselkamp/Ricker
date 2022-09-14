#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 12:50:23 2022

@author: Marieke_Wesselkamp
"""
import numpy as np
import matplotlib.pyplot as plt

s = np.linspace(-200,1, 5000)

def softplus(z):
    theta = np.log(1 + np.exp(z))
    return theta

def x(sigma):
    x = -50000*np.log(softplus(sigma)) - 5/(softplus(sigma)**2)
    return x


plt.plot(s, x(s))
