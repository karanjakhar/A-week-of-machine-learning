#loading required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#loading data for classification 
df = pd.read_csv('https://query.data.world/s/67p5gkjye5vocfiqm2cuxnrkx4ijim')

#looking at first five rows
df.head()

#checking details about the data
df.info()

#filling missing values
df['3P%'].fillna(0,inplace = True)

#Getting target variable and features in different variables
y_train = df['TARGET_5Yrs']
x_train = df.drop(['Name','TARGET_5Yrs'],axis = 1)

#spliting the data into train and test set
x_train,x_test,y_train,y_test = train_test_split(x_train,y_train)

#fit data in our model and check the result
clf = SVC()
clf.fit(x_train,y_train)
print('Accuracy:',clf.score(x_test,y_test))
