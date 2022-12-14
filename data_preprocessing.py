# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 16:31:28 2022

@author: thiag
"""
#importing libraries
import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn.impute import SimpleImputer

#importing the dataset
dataset = pd.read_csv("Data/Data.csv")

#spliting the dataset into dependent and independent variables
X = dataset.iloc[: , :-1].values #Todas as linhas, menos a última coluna
y = dataset.iloc[: , -1].values  #Última linha, última coluna
print('X spliting dataset: ',X,sep='\n')
#taking care of missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:,1:3])
print('X taking care of missing data: ',X,sep='\n')

#enconding categorical data#####################################################################
#It transforms the name strings into numbers    
#independent variable:                                               # 
from sklearn.compose import ColumnTransformer                                                  #
from sklearn.preprocessing import OneHotEncoder                                                #
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough') #
X  = np.array(ct.fit_transform(X))                                                             #
print('encoding categorical data: ',X,sep='\n')                                                #
################################################################################################

#encoding the dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y  = le.fit_transform(y)
print(y)

#spliting data into Training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= 0.2,random_state= 1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)


