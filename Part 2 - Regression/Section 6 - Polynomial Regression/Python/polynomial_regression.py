# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 12:52:31 2023

@author: adina
"""

import pandas as pd 
import numpy as np
import matplotlib as plt 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

dataset = pd.read_csv('Position_Salaries.csv')
X= dataset.iloc[: , 1:-1]
y = dataset.iloc[: , -1]

regressor  = LinearRegression()
regressor.fit(X,y)

poly_reg = PolynomialFeatures(degree=2)
X_poly= poly_reg.fit_transform(X)



regressor_2 = LinearRegression()
regressor_2.fit(X_poly, y)

y_pred = regressor_2.predict(poly_reg.transform([[5.5]]))

print("the prediction of 5.5 is:", y_pred)
