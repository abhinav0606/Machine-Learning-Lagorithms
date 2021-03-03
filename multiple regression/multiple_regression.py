# importing the modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("/home/abhinav/Documents/Machine Learning/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 5 - Multiple Linear Regression/Python/50_Startups.csv")
print(dataset)
X=dataset.iloc[:,0:4]
Y=dataset.iloc[:,-1]
print(X)
print(Y)
X=X.values
Y=Y.values
# encoding the data values for that one column which contanins string
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[3])],remainder="passthrough")
X=np.array(ct.fit_transform(X))
print(X)


# training and splitting the dataset into training and test data set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
# X_train=X_train.reshape(-1,1)
# X_test=X_test.reshape(-1,1)
# Y_train=Y_train.reshape(-1,1)
# Y_test=Y_test.reshape(-1,1)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

# Now training the train dataset on multiple linear regression model

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
# predicting the X_test value
pred=lr.predict(X_test)
# set the precision
np.set_printoptions(precision=2)
z=np.concatenate((pred.reshape(len(pred),1),Y_test.reshape(len(Y_test),1)),1)
print(z)
prediction=[]
real=[]
for i in z:
    prediction.append(i[0])
    real.append(i[1])
a=list(range(1,len(prediction)+1))
plt.scatter(a,prediction,color="red")
plt.plot(a,real,color="blue")
plt.show()


# now as we want the single prediction so we get it as
# california=first
# newyork=last
# florida=2nd
print(lr.predict([[1,0,0,20000,25,50]]))

# coefficient
print(lr.coef_)
# intercept
print(lr.intercept_)