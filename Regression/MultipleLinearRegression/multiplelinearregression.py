# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 19:29:53 2022

@author: thiag
"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#PREPROCESSING DATA
#importing dataset
dataset = pd.read_csv("../../Data/50_Startups.csv")
X = dataset.iloc[: , :-1].values
y = dataset.iloc[: ,  -1].values

print(X,'\n')

#Encoding categorical data
from sklearn.compose       import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer (transformers=[('encoder',OneHotEncoder(),[3])], remainder='passthrough' )
X  = np.array(ct.fit_transform(X))

print(X)

#Spliting the dataset
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=0)

#Training the Multiple Linear Regression Model on the train set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train , y_train)

#Predicting test results 
y_predicted = regressor.predict(X_test)
np.set_printoptions(precision=(2))             
print(np.concatenate( (y_predicted.reshape(len(y_predicted),1) , y_test.reshape(len(y_test),1)) , 1))

