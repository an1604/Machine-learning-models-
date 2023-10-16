# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 13:05:40 2023

@author: adina
"""

import pandas as pd 
import numpy as np
import matplotlib as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv('Position_Salaries.csv')
X= dataset.iloc[:,1:-1].values
y= dataset.iloc[:,-1].values

sc= StandardScaler()
X= sc.fit_transform(X)

sc1 = StandardScaler()
y= sc1.fit_transform(y.reshape(-1,1))


regressor= SVR(kernel='rbf')
regressor.fit(X, y)

y_pred = regressor.predict(sc.transform([[5.5]]).reshape(-1,1))

print("The prediction of 5.5 years exp is : " , sc1.inverse_transform([y_pred]))