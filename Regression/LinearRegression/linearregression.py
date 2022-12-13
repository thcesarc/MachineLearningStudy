# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 00:26:47 2022

@author: thiag
"""

#libraries
#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv("../../Data/Salary_Data.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

#spliting dataset into test and training
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=1/3, random_state=0)

#Training the Simple Linear Regression Model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)#prediction

y_pred = regressor.predict(X_test) #predicted
#training results
plt.scatter(X_train , y_train , color='red') #tipo do gráfico e o que ele recebe para grafar
plt.plot(X_train , regressor.predict(X_train), color='black')

plt.title("Salary vs Experience (Training set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")

plt.show()


#test results
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regressor.predict(X_train),color='black')

plt.title("Salary vs Experience (Test set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")

plt.show()

