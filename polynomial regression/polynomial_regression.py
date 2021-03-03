import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("/home/abhinav/Documents/Machine Learning/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 6 - Polynomial Regression/Python/Position_Salaries.csv")
print(dataset)
X=dataset.iloc[:,1:-1].values
Y=dataset.iloc[:,-1].values
print(X)
print(Y)
# as data is very less do we will just miss the splitting the data
# into training and testing sets
# Training the sets in linear _regresion
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X,Y)
print(lr.predict(X))

# creating a polynomial_regression
from sklearn.preprocessing import PolynomialFeatures
ploy_reg=PolynomialFeatures(degree=5)
X_poly=ploy_reg.fit_transform(X)
lr2=LinearRegression()
lr2.fit(X_poly,Y)
print(lr2.predict(X_poly))
print(lr2.predict(ploy_reg.fit_transform([[6.5]])))

# linear_plot
plt.scatter(X,Y,color="red")
plt.plot(X,lr.predict(X),color="blue")
plt.show()

plt.scatter(X,Y,color="red")
plt.plot(X,lr2.predict(ploy_reg.fit_transform(X)),color="blue")
plt.show()

# plotting very closely
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color="red")
plt.plot(X_grid,lr2.predict(ploy_reg.fit_transform(X_grid)),color="blue")
plt.show()