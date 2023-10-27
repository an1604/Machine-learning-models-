# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 13:21:26 2023

@author: adina
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the dataset (corrected filename)
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Print the size of the dataset (removed the parentheses)
print(dataset.size)

corpus = []
all_stopwords = stopwords.words('english')
for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
    review=review.lower()
    review=review.split()
    ps = PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(all_stopwords)]
    review=' '.join(review)
    corpus.append(review)

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y= dataset.iloc[: , -1]

X_train , X_test , y_train , y_test = train_test_split(X, y , test_size=0.2 , random_state=0)
classidier = GaussianNB()
classidier.fit(X_train,y_train)
y_pred= classidier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
    

