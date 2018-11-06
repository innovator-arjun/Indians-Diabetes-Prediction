# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:48:03 2018

@author: ARJUN_PC
"""

#https://www.kaggle.com/uciml/pima-indians-diabetes-database

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample

dataset=pd.read_csv('diabetes.csv')

dataset.Pregnancies.unique()
dataset.isnull().sum()





dataset.columns
print((dataset[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] == 0).sum())
dataset.describe()
dataset[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = dataset[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0, np.NaN)


#dataset.dropna(inplace=True)



from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values ='NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(dataset[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']])
dataset[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]= imputer.transform(dataset[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']])




X=dataset.iloc[:,:8].values

y=dataset.iloc[:,8].values
dataset.describe()


#Categorical Variablehence we use label encoder 
from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()

dataset['Pregnancies']=label.fit_transform(dataset['Pregnancies'])






#Splitting independent and dependent variable
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

X_train

df_majority = dataset[dataset.Outcome==0]
df_minority = dataset[dataset.Outcome==1]

print(df_majority.Outcome.count())
print("-----------")
print(df_minority.Outcome.count())
print("-----------")
print(dataset.Outcome.value_counts())

#UPSAMPLING DATA

df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=500,    # to match majority class
                                 random_state=587) # reproducible results
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
# Display new class counts
df_upsampled.Outcome.value_counts()

df_upsampled
x_up = df_upsampled.drop(['Outcome'],axis = 1)
y_up = df_upsampled.Outcome


# Further divide the train data into train test 
X_train_up, X_test_up, y_train_up, y_test_up = train_test_split(x_up, y_up, test_size=0.25, random_state=2)


#Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_up = sc.fit_transform(X_train_up)
X_test_up = sc.transform(X_test_up)

from sklearn.decomposition import PCA
pca=PCA(n_components=None,random_state=0)
X_train_up=pca.fit_transform(X_train_up)
X_test_up=pca.transform(X_test_up)
explained_variance=pca.explained_variance_ratio_



from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100)
classifier.fit(X_train_up,y_train_up)

y_pred=classifier.predict(X_test_up)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_pred,y_test_up)



#Applying the k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
#10 accuracy will be returned that will be computed through 10 computation using k-fold
accuracies=cross_val_score(estimator=classifier,X=X_train_up,y=y_train_up,cv=10)

#Take the average or mean on accuracies
mean_accuracies=accuracies.mean()
std_accuracies=accuracies.std()*100


from matplotlib.colors import ListedColormap
X_set, y_set = X_train_up, y_train_up
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

210/250











