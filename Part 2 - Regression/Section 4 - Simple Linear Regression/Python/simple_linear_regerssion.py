# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 12:06:56 2023

@author: adina
"""
import numpy as np
import pandas as pd 
import matplotlib as plt 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



"import the dataset : "

dataset = pd.read_csv('Salary_Data.csv')
X= dataset.iloc[: , :-1].values
y = dataset.iloc[: , -1].values
 
"seperate to train set and test set : "
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=0)

"model : "
regressor = LinearRegression()
regressor.fit(X_train,y_train)

"prediction:"

y_pred = regressor.predict(X_test)

print('x test is : ' , X_test)

print('The predictions are : ',y_pred)