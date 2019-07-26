#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:09:00 2019

@author: shreyas
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import re
import os
import nltk
nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

reviews_train=[]
for line in open('movie_data/full_train.txt', 'r'):
    
    reviews_train.append(line.strip())
    
reviews_test=[]
for line in open('movie_data/full_test.txt', 'r'):
    
    reviews_test.append(line.strip())

target = [1 if i<12500 else 0 for i in range(25000)]

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess(reviews):
    reviews = [REPLACE_NO_SPACE.sub('', line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(' ', line.lower()) for line in reviews]
    
    return reviews

reviews_train = preprocess(reviews_train)
reviews_test = preprocess(reviews_test)

from nltk.corpus import stopwords

english_stop_words = ['in', 'of', 'at', 'a', 'the', 'an'] 
def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split() 
                      if word not in english_stop_words])
        )
    return removed_stop_words

reviews_train = remove_stop_words(reviews_train)
reviews_test = remove_stop_words(reviews_test)

def lemmatize(corpus):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_text=[]
    
    for review in corpus:
        lemmatized_text.append(
                ' '.join([lemmatizer.lemmatize(word) for word in review.split()])
                )
    
    return lemmatized_text

reviews_train = lemmatize(reviews_train)
reviews_test = lemmatize(reviews_test)

cv = CountVectorizer(binary=False, ngram_range=(1, 3)) #can also use tfidf vectorizer instead
cv.fit(reviews_train)
X = cv.transform(reviews_train)
X_test = cv.transform(reviews_test)

X_train, X_val, y_train, y_val = train_test_split(
    X, target, train_size = 0.75
)

for c in [0.001, 0.005, 0.01, 0.05, 0.1]:
    sv=LinearSVC(C=c)
    sv.fit(X_train, y_train)
    print (" Train Accuracy for C=%s: %s"
           % (c, accuracy_score(y_val, sv.predict(X_val)))) 

    
svm=LinearSVC(C=0.01)
svm.fit(X, target)
print (" Test Accuracy: %s"
           % (accuracy_score(target, svm.predict(X_test)))) 
