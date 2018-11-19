# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:05:24 2018

@author: avaithil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('BlackFriday.csv')

dataset.describe()

dataset.info

dataset.head()

dataset.shape



dataset.info()
dataset=dataset.fillna('NaN')

X=dataset.iloc[:,2:11].values
y=dataset.iloc[:,11:12].values

#To fill with missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='median',axis=0)
imputer = imputer.fit(X[:, 7:9])
X[:, 7:9] = imputer.transform(X[:, 7:9])



from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Label Encoding for columns
labelencoder_X1 = LabelEncoder()
X[:, 0] = labelencoder_X1.fit_transform(X[:, 0])


labelencoder_X2 = LabelEncoder()
X[:, 1] = labelencoder_X2.fit_transform(X[:, 1])

labelencoder_X3 = LabelEncoder()
X[:, 3] = labelencoder_X3.fit_transform(X[:, 3])

labelencoder_X4 = LabelEncoder()
X[:, 4] = labelencoder_X4.fit_transform(X[:, 4])

X[0:9]

8,13,16


#To create dummy variable for the category
onehotencoder = OneHotEncoder(categorical_features=[0,1,3,4])

X = onehotencoder.fit_transform(X).toarray()
X.shape

X=np.delete(X,[8,13,16],axis=1)



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#c_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)


#https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/
#metrics to evaluate the model
from sklearn.ensemble import RandomForestRegressor

regressor=RandomForestRegressor(n_estimators=100)

regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

from sklearn import model_selection

scoring = 'r2'
results=model_selection.cross_val_score(regressor,X,y,scoring=scoring)
mean=results.mean()*100
std=results.std()*100






"""

dataset.Product_Category_3.value_counts().count()

dataset.isnull().sum()

gender_map={'M':0,'F':1}
dataset['Gender']=dataset['Gender'].map(gender_map)


city_map={'A':0,'B':1,'C':2}
dataset['City_Category']=dataset['City_Category'].map(city_map)


dataset.Age.value_counts()


age_map={'0-17':0,'18-25':1,'26-35':2,'36-45':3,'46-50':4,'51-55':5,'55+':6}
dataset['Age']=dataset['Age'].map(age_map)


dataset.Gender=dataset.Gender.astype('category')

dataset.Age=dataset.Age.astype('category')

dataset.info()"""