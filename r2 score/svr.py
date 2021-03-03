import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("/home/abhinav/Documents/Machine Learning A-Z (Model Selection)/Regression/Data.csv")
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
Y=Y.reshape(len(Y),1)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)
sc_y=StandardScaler()
Y_train=sc_y.fit_transform(Y_train)
Y_test=sc_y.transform(Y_test)
from sklearn.svm import SVR
svr=SVR(kernel="rbf")
svr.fit(X_train,Y_train)
print(sc_y.inverse_transform(svr.predict(X_test)))
y_pred=sc_y.inverse_transform(svr.predict(X_test))
Y_test=sc_y.inverse_transform(Y_test)
np.set_printoptions(precision=2)
z=np.concatenate((y_pred.reshape(len(y_pred),1),Y_test.reshape(len(Y_test),1)),1)
print(z)
from sklearn.metrics import r2_score
print(r2_score(Y_test,y_pred))