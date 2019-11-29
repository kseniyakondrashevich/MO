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

mat = loadmat('data/ex9_movies.mat')
R = mat['R']
Y = mat['Y']

NUM_FEATURES = 15

class CollaborativeFiltering:
    def __init__(self, num_features=NUM_FEATURES, gradient_step=0.5, reg_lambda=0.1, max_iters=5000):
        self.num_features = num_features
        self.gradient_step = gradient_step
        self.reg_lambda = reg_lambda
        self.max_iters = max_iters
    
    def cost_func(self, Y, R):
        hypotesis = np.dot(self.X, self.Theta)
        mean_error = R * (hypotesis - Y)
        mean_squared_error = mean_error ** 2
        cost = mean_squared_error.sum() / 2
        regularized_cost = cost + (self.reg_lambda / 2) * ((self.X ** 2).sum() + (self.Theta ** 2).sum())
        return regularized_cost

    def gradient_descent(self, Y, R):
        hypotesis = np.dot(self.X, self.Theta)
        mean_error = R * (hypotesis - Y)
        dX = np.dot(mean_error, self.Theta.T)
        dTheta = np.dot(self.X.T, mean_error)
        regularized_dX = dX + self.reg_lambda * self.X
        regularized_dTheta = dTheta + self.reg_lambda * self.Theta
        self.X -= self.gradient_step * regularized_dX
        self.Theta -= self.gradient_step * regularized_dTheta

    def fit(self, Y, R):
        self.n_m, self.n_u = Y.shape 
        self.X = np.random.rand(self.n_m, self.num_features)
        self.Theta = np.random.rand(self.num_features, self.n_u)

        for cur_step in range(self.max_iters):
            self.gradient_descent(Y, R)
            cost = self.cost_func(Y, R)

    def predict(self, user_id, R, top=5):
        predictions = np.dot(self.X, self.Theta)
        user_ratings = (R[:, user_id] != 1) * predictions[:, user_id]
        return user_ratings.argsort()[-top:][::-1]class CollaborativeFiltering:
    def __init__(self, num_features=NUM_FEATURES, gradient_step=0.5, reg_lambda=0.1, max_iters=5000):
        self.num_features = num_features
        self.gradient_step = gradient_step
        self.reg_lambda = reg_lambda
        self.max_iters = max_iters
    
    def cost_func(self, Y, R):
        hypotesis = np.dot(self.X, self.Theta)
        mean_error = R * (hypotesis - Y)
        mean_squared_error = mean_error ** 2
        cost = mean_squared_error.sum() / 2
        regularized_cost = cost + (self.reg_lambda / 2) * ((self.X ** 2).sum() + (self.Theta ** 2).sum())
        return regularized_cost

    def gradient_descent(self, Y, R):
        hypotesis = np.dot(self.X, self.Theta)
        mean_error = R * (hypotesis - Y)
        dX = np.dot(mean_error, self.Theta.T)
        dTheta = np.dot(self.X.T, mean_error)
        regularized_dX = dX + self.reg_lambda * self.X
        regularized_dTheta = dTheta + self.reg_lambda * self.Theta
        self.X -= self.gradient_step * regularized_dX
        self.Theta -= self.gradient_step * regularized_dTheta

    def fit(self, Y, R):
        self.n_m, self.n_u = Y.shape 
        self.X = np.random.rand(self.n_m, self.num_features)
        self.Theta = np.random.rand(self.num_features, self.n_u)

        for cur_step in range(self.max_iters):
            self.gradient_descent(Y, R)
            cost = self.cost_func(Y, R)

    def predict(self, user_id, R, top=5):
        predictions = np.dot(self.X, self.Theta)
        user_ratings = (R[:, user_id] != 1) * predictions[:, user_id]
        return user_ratings.argsort()[-top:][::-1]