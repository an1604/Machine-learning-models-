# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:39:13 2023

@author: adina
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

N, d = dataset.shape
ads_selected = []
numbers_of_rewards1 = [0] * d
numbers_of_rewards0 = [0] * d
total_reward = 0

for n in range(N):
    ad = 0
    max_random = 0
    
    for i in range(d):
        random_beta = random.betavariate(numbers_of_rewards1[i] + 1, numbers_of_rewards0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    
    if reward == 1:
        numbers_of_rewards1[ad] += 1
    else:
        numbers_of_rewards0[ad] += 1
    
    total_reward += reward


max_ad = max(numbers_of_rewards1) 
index =0
for i in range(numbers_of_rewards1.__len__()):
    if max_ad==numbers_of_rewards1[i]:
        index = i
        break
    
print('Total reward : ' ,index )
print('with : ' ,max_ad )

