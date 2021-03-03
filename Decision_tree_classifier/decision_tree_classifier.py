import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("/home/abhinav/Documents/Machine Learning/Machine Learning A-Z (Codes and Datasets)/Part 3 - Classification/Section 18 - Naive Bayes/Python/Social_Network_Ads.csv")
print(dataset)
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
print(X)
print(Y)
# splitting results
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
# feature scalling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.tree import DecisionTreeClassifier
dct=DecisionTreeClassifier(criterion="entropy",random_state=0)
dct.fit(X_train,Y_train)
print(dct.predict(sc.transform([[50,60000]])))
y_pred=dct.predict(X_test)
z=np.concatenate((y_pred.reshape(len(y_pred),1),Y_test.reshape(len(Y_test),1)),1)
print(z)
from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(Y_test,y_pred))
print(confusion_matrix(Y_test,y_pred))