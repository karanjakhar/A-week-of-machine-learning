#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#loading data into dataframe
df = pd.read_csv('https://query.data.world/s/67p5gkjye5vocfiqm2cuxnrkx4ijim')

#printig first five rows
df.head()

#getting basic detail
df.info()

#filling missing values
df['3P%'].fillna(0,inplace = True)

#checking data balance
df['TARGET_5Yrs'].value_counts().plot.bar()

#getting target and features in different variables
y_train = df['TARGET_5Yrs']
X_train = df.drop(['TARGET_5Yrs','Name'],axis = 1)

#mean = 0 and std = 1
X_train = StandardScaler().fit_transform(X_train)

#splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)

#different classifiers
clfs = [LogisticRegression(), DecisionTreeClassifier(), ExtraTreeClassifier(), RandomForestClassifier(), SVC(), GaussianNB(), KNeighborsClassifier()]
c_names = ['Logistic Regression','Decision Tree', 'Extra Tree', 'Random Forest', 'SVC', 'Naive bayes', 'KNN']

#fitting all the classifier
res = {}
for c_name,clf in zip(c_names,clfs):
  clf.fit(X_train,y_train)
  acc = clf.score(X_test,y_test)
  res[c_name] = acc
  print('{} : {}'.format(c_name,acc))

 print(res)
