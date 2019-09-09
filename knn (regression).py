#importing required libraries
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#loading data for regression
r_df = pd.read_csv('boston_train.csv')

#printing first five rows
r_df.head()

#getting basic details
r_df.info()

#getting our target and features in different variable
y_train = r_df['medv']
X_train = r_df.drop(['medv','ID'],axis = 1)

#splitting data into train and test sets
X_train,X_test,y_train,y_test = train_test_split(X_train,y_train)

#train and test the model
reg = KNeighborsRegressor()
reg.fit(X_train, y_train)
print('Error:',mean_squared_error(reg.predict(X_test), y_test))
