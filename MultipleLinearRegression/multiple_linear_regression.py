#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:54:37 2018

@author: 738456
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

#Avoiding the dummy variable trap
x = x[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the test set results
y_pred = regressor.predict(x_test)

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
x = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis = 1)

x_opt = x[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

def backwardElimination(x, sl):
    numVars = len(x[0])
    print(numVars)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        print(maxVar)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if(regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    print(regressor_OLS.summary())
    return x

SL = 0.05
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
x_opt_model = backwardElimination(x_opt, SL)

np.hstack

#x_opt = x[:, [0, 1, 3, 4, 5]]
#regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
#regressor_OLS.summary()
#
#x_opt = x[:, [0, 3, 4, 5]]
#regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
#regressor_OLS.summary()
#
#x_opt = x[:, [0, 3, 5]]
#regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
#regressor_OLS.summary()
#
#x_opt = x[:, [0, 3]]
#regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
#regressor_OLS.summary()



