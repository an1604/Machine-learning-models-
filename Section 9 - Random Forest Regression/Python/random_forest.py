# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 14:19:33 2023

@author: adina
"""

import pandas as pd 
import numpy as np
import matplotlib as plt 
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv('Position_Salaries.csv')
X= dataset.iloc[: , 1:-1].values
y = dataset.iloc[: , -1].values

regressor= RandomForestRegressor(n_estimators=10 , random_state= 0 )
regressor.fit(X,y)

y_pred = regressor.predict([[5.5]])
print("the pridicted value of 5.5 is : " , y_pred)