#importing required libraries
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

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

#splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)

#creating an instance
clf = lgb.LGBMClassifier()

#training the classifier and testing
clf.fit(X_train,y_train)
acc = clf.score(X_test,y_test)
print(acc)
