# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 15:38:20 2022

@author: thiag
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("../../Data/Position_Salaries.csv")
X = dataset.iloc[: , 1:-1].values
y = dataset.iloc[: , -1].values

########################Linear Regression Training
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
##visualization
plt.scatter(X,y,color="red")
plt.plot(X,lin_reg.predict(X), color ="black")


#########################PolynomialRegression Training
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly , y)
#visualizing
plt.scatter(X,y, color="red")
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)),color="blue")


