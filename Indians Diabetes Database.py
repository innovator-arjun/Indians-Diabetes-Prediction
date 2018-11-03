# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:48:03 2018

@author: ARJUN_PC
"""

#https://www.kaggle.com/uciml/pima-indians-diabetes-database

import pandas as pd


dataset=pd.read_csv('diabetes.csv')

dataset.Pregnancies.unique()


#Categorical Variablehence we use label encoder 
from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()

dataset['Pregnancies']=label.fit_transform(dataset['Pregnancies'])


#Splitting independent and dependent variable
X=dataset.iloc[:,0:8].values

y=dataset.iloc[:,8:9].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X=sc.fit_transform(X)




from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()
classifier.fit(X)