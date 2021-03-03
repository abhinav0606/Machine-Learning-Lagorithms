import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# data preprocessing techniques
data=pd.read_csv("/home/abhinav/Documents/Machine Learning/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 4 - Simple Linear Regression/Python/Salary_Data.csv")
X=data.iloc[:,0].values
Y=data.iloc[:,1].values
print(X)
print(Y)
# training the dataset and splitting it into training and dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
# training the train set into the linear regression model
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
X_train=X_train.reshape(-1,1)
Y_train=Y_train.reshape(-1,1)
X_test=X_test.reshape(-1,1)
lr.fit(X_train,Y_train)
# creating a predicting set on X_test
prediction=lr.predict(X_test)
# plotting the plots
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,lr.predict(X_train),color="blue")
plt.title("Experience vs salary")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
plt.scatter(X_test,Y_test,color="red")
plt.plot(X_test,prediction,color="blue")
plt.title("Experience vs salary")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
# predicting the salary according to the experience
print(lr.predict([[50]]))
# finding the equation
print(lr.coef_)
print(lr.intercept_)