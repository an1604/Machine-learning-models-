# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:44:11 2023

@author: adina
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori


def inspect(results):
    lhs = [tuple(result[2][0][0][0]) for result in results]
    rhs = [tuple(result[2][0][1][0]) for result in results]
    supports= [result[1] for result in results]
    confidences= [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs,rhs,supports,confidences,lifts))






dataset = pd.read_csv('Market_Basket_Optimisation.csv' , header=None)

transactions = [] 

for i in range(0, dataset.shape[0]):
    transaction = []
    for j in range(0, dataset.shape[1]):
        item = str(dataset.values[i, j])
        if item != 'nan':  # Exclude NaN values
            transaction.append(item)
    transactions.append(transaction)
   

rules = apriori(
    transactions=transactions,
    min_support=0.003,
    min_confidence=0.2,
    min_lift=3,
    min_length=2,
    max_length=2
)

result = list(rules)

resultsinDF = pd.DataFrame(inspect(result),columns=['Left Hand Side' , 'Right Hans Side' , 'Support' , 'Confidence' , 'Lift'])
resultsinDF
