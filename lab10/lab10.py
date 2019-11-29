import copy
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from scipy import misc
from datetime import datetime
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
import scipy.optimize
from scipy import stats
from sklearn.tree import *
from sklearn.ensemble import *


import math
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_boston
boston_dataset = load_boston()
X = boston_dataset["data"]
y = boston_dataset["target"]

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=42)

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

def grad(z, y):
    return y - z

def gbm_predict(X, base_algorithms_list, coefficients_list):
    return [sum([coeff * algo.predict([x])[0] for algo, coeff in zip(base_algorithms_list, coefficients_list)]) for x in X]

def gbm_train(X_train, y_train, max_depth=5, iters=50, coef=0.9):
    algos, coefs = [], []
    
    target = y_train
    for i in range(iters):
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        tree.fit(X_train, target)
        algos.append(tree)
        
        if isinstance(coef, (int, float)):
            coefs.append(coef)
        else:
            coefs.append(coef(i))
            
        target = grad(gbm_predict(X_train, algos, coefs), y_train)
        
    return algos, coefs

    
algos, coefs = gbm_train(X_train, y_train)
mean_squared_error(y_test, gbm_predict(X_test, algos, coefs))

algos, coefs = gbm_train(X_train, y_train)
mean_squared_error(y_test, gbm_predict(X_test, algos, coefs))