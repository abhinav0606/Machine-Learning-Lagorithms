import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# as we dont have to do so many things for the preprocessing
# as no feature scalling

dataset=pd.read_csv("/home/abhinav/Documents/Machine Learning/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 8 - Decision Tree Regression/Python/Position_Salaries.csv")
X=dataset.iloc[:,1:-1].values
Y=dataset.iloc[:,-1].values
print(X)
print(Y)
# as nopt feature scalling and splitting bcz dataset is so small
# at low level analyzing the dataset doesnt mean anything
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor(random_state=0)
dt.fit(X,Y)
print(dt.predict([[6.50]]))
# plotting the graph and visualising it

plt.scatter(X,Y,color="red")
plt.plot(X,dt.predict(X),color="blue")
plt.show()




# making it at higher resolution

X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color="red")
plt.plot(X_grid,dt.predict(X_grid),color="blue")
plt.show()


print(dt.predict(X_grid))