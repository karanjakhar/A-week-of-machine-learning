#Importing required libraries
from sklearn.linear_model import  LinearRegression
from sklearn.datasets import california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Downloading dataset
data = california_housing.fetch_california_housing()

#Getting target and features
train_data = data['data']
target = data['target']
description = data['DESCR']
feature_names = data['feature_names']

#Features present in our data 
print(feature_names)

#Our Training data
print(train_data)

#Type of our training data
print(type(train_data))

#our target values
print(target)

#type of our target values
print(type(target))

#splitting our data in train and validation set
x_train,x_test,y_train,y_test = train_test_split(train_data,target)

#our model
model = LinearRegression()
model.fit(x_train,y_train)
print(mean_squared_error(model.predict(x_test),y_test))
