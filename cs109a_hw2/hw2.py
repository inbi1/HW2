# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:35:31 2019

@author: inbi
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.api import OLS


df = pd.read_csv('G:/My Drive/2019-CS109A_HW/cs109a_hw2/data/nyc_taxi.csv') 

X = df.iloc[:, 0]
y = df.iloc[:, 1]
T = np.linspace(0, max(X), 1000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

plt.scatter(X_train, y_train)
plt.title('Time vs. Trips')
plt.xlabel('Time of day')
plt.ylabel('Trips')

KNNModels = {}
n_neighbors = 5
knn = KNeighborsRegressor(n_neighbors, weights='distance')


