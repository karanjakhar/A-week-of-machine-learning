#importing libraries
from sklearn.linear_model import  LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

#loading data
data = pd.read_csv('path of your csv file here.csv')

#getting information 
data.info()

#printing first 5 rows of the data
data.head()

#getting target and features 
train_data = df.drop('target',axis = 1)
y_train = df['target']

#separating train and test set
x_train,x_test,y_train,y_test = train_test_split(train_data,y_train)

#training model and predicting 
model = LinearRegression()
model.fit(x_train,y_train)
print(mean_squared_error(model.predict(x_test),y_test))
