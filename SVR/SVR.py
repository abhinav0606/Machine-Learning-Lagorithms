import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("/home/abhinav/Documents/Machine Learning/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/Python/Position_Salaries.csv")
X=dataset.iloc[:,1:-1].values
Y=dataset.iloc[:,-1].values
print(X)
print(Y)

# as now we have no missing data and the dataset is too small so we cannot do any kind of splitting
# but as in svr we dont have any kind of specific equation
# so we have to apply feature scalling ad feature scalling need an unique structure as
# x is 2d and y is 1d so we are changing y to 2d
Y=Y.reshape(-1,1)

# so for x and y we have to create seperate objects
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
X=sc_x.fit_transform(X)
Y=sc_y.fit_transform(Y)
print(X)
print(Y)

# modelling into the svm machine with the help of SVR
from sklearn.svm import SVR
Y=Y.flatten()
reg=SVR(kernel="rbf")
reg.fit(X,Y)
# now as we have scalled it so we have to transform it now so we have to use another
# method to do it
print(sc_y.inverse_transform(reg.predict(sc_x.transform([[10]]))))
# we have to keep in mind about the scalling where we have to inverse and where we have
# to transform
# for normal first transform and to make it normal do inverse

# plotting the plot

plt.scatter(sc_x.inverse_transform(X),sc_y.inverse_transform(Y),color="red")
plt.plot(sc_x.inverse_transform(X),sc_y.inverse_transform(reg.predict(X)),color="blue")
plt.show()

X_grid=np.arange(min(sc_x.inverse_transform(X)),max(sc_x.inverse_transform(X)),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(sc_x.inverse_transform(X),sc_y.inverse_transform(Y),color="red")
plt.plot(X_grid,sc_y.inverse_transform(reg.predict(sc_x.transform(X_grid))),color="blue")
plt.show()