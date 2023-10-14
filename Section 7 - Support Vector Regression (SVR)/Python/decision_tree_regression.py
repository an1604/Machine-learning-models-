# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 13:17:48 2023

@author: adina
"""


import pandas as pd 
import numpy as np
import matplotlib as plt 
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv('Position_Salaries.csv')
X= dataset.iloc[: , 1:-1].values
y = dataset.iloc[: , -1].values

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

print(regressor.predict([[9.5]]))

 