# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 14:48:00 2023

@author: adina
"""


import pandas as pd 
import numpy as np
import matplotlib as plt 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv('Social_Network_Ads.csv')
X= dataset.iloc[: , 1:-1].values
y = dataset.iloc[: , -1].values

X_train , X_test , y_train , y_test = train_test_split(X,y , test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifer= GaussianNB()
classifer.fit(X_train,y_train)


y_pred = classifer.predict(X_test)

# get the predictions one against other...
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

# get the confusion matrix and accuracy rate : 
cm = confusion_matrix(y_test, y_pred)
print('matrix: ',cm)
print( 'rate : ',accuracy_score(y_test, y_pred))