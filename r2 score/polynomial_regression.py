import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("/home/abhinav/Documents/Machine Learning A-Z (Model Selection)/Regression/Data.csv")
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
print(X)
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly=PolynomialFeatures(degree=5)
x_poly=poly.fit_transform(X_train)
lr=LinearRegression()
lr.fit(x_poly,Y_train)
print(lr.predict(poly.transform(X_test)))
y_pred=lr.predict(poly.transform(X_test))
np.set_printoptions(precision=2)
z=np.concatenate((y_pred.reshape(len(y_pred),1),Y_test.reshape(len(Y_test),1)),1)
print(z)
from sklearn.metrics import r2_score
print(r2_score(Y_test,y_pred))