# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# Load the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# Separate features and target variable
X = dataset.iloc[:, :2].values
y = dataset.iloc[:, -1].values

# Apply standardization to numerical features (assuming columns 1 and 2 are numerical)
sc = StandardScaler()
X[:, 1:] = sc.fit_transform(X[:, 1:])

classifier = GaussianNB()
classifier.fit (X,y)

accuracies = cross_val_score(estimator= classifier, X=X, y=y , cv=10)
print('Accuracy :{:.2f}%'.format(accuracies.mean()*100))
print('Standart Deriation:{:.2f}%'.format(accuracies.std()*100))
