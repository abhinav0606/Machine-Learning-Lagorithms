import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("/home/abhinav/Documents/Machine Learning/Machine Learning A-Z (Codes and Datasets)/Part 3 - Classification/Section 14 - Logistic Regression/Python/Social_Network_Ads.csv")
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
print(X)
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
print(X_train)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=0)
lr.fit(X_train,Y_train)
print(lr.predict(X_test))

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=10,metric="minkowski",p=2)
knn.fit(X_train,Y_train)
print(knn.predict(X_test))

from sklearn.svm import SVC
svc=SVC(kernel="linear",random_state=0)
svc.fit(X_train,Y_train)
print(svc.predict(X_test))

from sklearn.svm import SVC
kernel=SVC(kernel="rbf",random_state=0)
kernel.fit(X_train,Y_train)
print(kernel.predict(X_test))
