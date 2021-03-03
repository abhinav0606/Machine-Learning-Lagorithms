import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("/home/abhinav/Documents/Machine Learning/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 9 - Random Forest Regression/Python/Position_Salaries.csv")
print(dataset)
X=dataset.iloc[:,1:-1].values
Y=dataset.iloc[:,-1].values
from sklearn.ensemble import RandomForestRegressor
rdnf=RandomForestRegressor(n_estimators=10,random_state=0)
rdnf.fit(X,Y)
print(rdnf.predict([[6.5]]))
print(max(X))
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color="red")
plt.plot(X_grid,rdnf.predict(X_grid))
plt.show()