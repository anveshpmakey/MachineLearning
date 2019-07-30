#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 15:14:20 2018

@author: Anvesh Makey
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train)

#Predicting the test set results
Y_pred = lr.predict(X_test)

#Visualising the training set results
plt.scatter(X_train, Y_train, color = 'green')
plt.plot(X_train, lr.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#Visualising the test set results
plt.scatter(X_test, Y_test, color = 'green')
plt.plot(X_test, lr.predict(X_test), color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()