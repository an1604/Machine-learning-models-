# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# Separate features and target variable
X = dataset.iloc[:, :2].values
y = dataset.iloc[:, -1].values

# Standardize the features
sc = StandardScaler()
X = sc.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the SVM classifier
classifier = SVC()

# Define the parameter grid for SVM (C parameter and kernel)
parameters = [{
    'C': [0.25, 0.5, 0.75],
    'kernel': ['linear']
},
    {
     'C': [0.25, 0.5, 0.75],
     'kernel': ['rbf'],
     'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  
     }
]


# Perform a grid search for hyperparameter tuning
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best accuracy and best parameters
best_accuracy = grid_search.best_score_
best_params = grid_search.best_params_

print('Accuracy: {:.2f}%'.format(best_accuracy * 100))
print('Best parameters:', best_params)
