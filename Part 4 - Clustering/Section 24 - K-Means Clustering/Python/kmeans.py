# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:25:07 2023

@author: adina
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

wcss =[]
size = dataset.size


for i in range(1,size+1):
    kmeans = KMeans(n_clusters= i , init='k-means++' , random_state=42 )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
kmeans = KMeans(n_clusters= 5 , init='k-means++' , random_state=42 )
kmeans.fit(X)


y_kmeans = kmeans.fit_predict(X)
print(y_kmeans)