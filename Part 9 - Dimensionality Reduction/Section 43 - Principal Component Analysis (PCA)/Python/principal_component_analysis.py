# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:47:21 2023

@author: adina
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print('The first line of X train : ' , X_train[0])

pca = PCA(n_components=2)
X_train=pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)

print('The first line of X train : ' , X_train[0])
