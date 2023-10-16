# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 12:30:50 2023

@author: adina
"""

import pandas as pd 
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

"dataset:"
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[: , :-1].values
y= dataset.iloc[: , -1].values


"Encoding : "
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


"splitting the data to train and test... " 
X_train, X_test , y_train , y_test = train_test_split(X, y , test_size= 0.2 , random_state=0) 

"model : "
regressor = LinearRegression()
regressor.fit(X_train,y_train)

X_new_test = [[1.0, 0.0, 0.0, 105671.96, 90790.61, 349744.55]]

"prediction: " 
y_pred = regressor.predict(X_new_test)

print("the new point is : " , X_new_test) 
print("The predction is : " , y_pred)
