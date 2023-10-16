# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:31:44 2023

@author: adina
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch




dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values


dendagram = sch.dendrogram(sch.linkage(X,method='ward'))

hc = AgglomerativeClustering(n_clusters=5 , affinity='euclidean' , linkage='ward')
