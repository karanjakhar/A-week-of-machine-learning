#loading required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

#loading data 
df = pd.read_csv('boston_train.csv')

#looking at first five rows
df.head()

#checking details about the data
df.info()

#Getting target variable and features in different variables
y_train = df['medv']
x_train = df.drop(['ID','medv'],axis = 1)

#spliting the data into train and test set
x_train,x_test,y_train,y_test = train_test_split(x_train,y_train)

#fit data in our model and check the error
reg = SVR()
reg.fit(x_train,y_train)
print('Error:',mean_squared_error(reg.predict(x_test),y_test))