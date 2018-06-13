from sklearn import tree
import pandas as pd
import numpy as np
import time

#loading the dataset for training
data_frame=pd.read_csv('train.csv')

#replacing male and female values with 0 and 1 respectively
data_frame['Sex'] = data_frame['Sex'].map({'female': 1, 'male': 0})
#print(data_frame['Sex'])
#removing those rows which do not have specific age
df_new=data_frame.dropna(subset=['Age'])

#tarining data for Age,sex and fare
train_data=df_new[['Age','Sex','Fare']]

#tarining target on the basis of survived or not
train_target=df_new[['Survived']]

#loading file for testing purpose
df_new_test=pd.read_csv('test.csv')
df_new_test['Sex'] = df_new_test['Sex'].map({'female': 1, 'male': 0})
df_new_test_mapped=df_new_test.dropna(subset=['Age','Sex','Fare'])
test_data=df_new_test_mapped[['Age','Sex','Fare']]
#test_target=df_new_test_mapped[['Survived']]
#Defining the decision tree classifier
clf=tree.DecisionTreeClassifier()

#training of classifier
trained=clf.fit(train_data,train_target)

#testing with the test data and test target
predicted=trained.predict(test_data)
#print(predicted)
new_col=pd.Series(predicted)
new_col_value=new_col.values
# print(len(new_col_value))
# print(len(df_new_test_mapped))

#assigning a new column named output
df_new_test_mapped = df_new_test_mapped.assign(output=new_col_value)

#saving the output onto a csv file
df_new_test_mapped.to_csv('check.csv', header=True)
