#importing required libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#loading data from a csv file to a pandas Dataframe
df = pd.read_csv('https://query.data.world/s/nsyvxagzhkssbiwytst5vpuvxpwgtb')

#looking at first 5 rows of the data
df.head()

#checking type of data and null values if any.
df.info()

#filling the null values
df['3P%'].fillna(0,inplace = True)

# getting our target variables and features in different variables
y_train = df['TARGET_5Yrs']
x_train = df.drop(['TARGET_5Yrs','Name'],axis = 1)

# spliting our data into train and test sets
x_train,x_test,y_train,y_test = train_test_split(x_train,y_train)

#training our classifier and check it's performance
clf = LogisticRegression()
clf.fit(x_train,y_train)
print('Accuracy:',clf.score(x_test,y_test))
