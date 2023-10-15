# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 14:38:42 2023

@author: adina
"""

import pandas as pd 
import numpy as np
import matplotlib as plt 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Social_Network_Ads.csv')
X= dataset.iloc[: , 1:-1].values
y = dataset.iloc[: , -1].values

X_train , X_test , y_train , y_test = train_test_split(X,y)

classifer = KNeighborsClassifier(n_neighbors=5 , metric='minkowski' , p=2)
classifer.fit(X_train, y_train)

y_pred = classifer.predict(X_test)

# get the predictions one against other...
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

# get the confusion matrix and accuracy rate : 
cm = confusion_matrix(y_test, y_pred)
print('matrix: ',cm)
print( 'rate : ',accuracy_score(y_test, y_pred))