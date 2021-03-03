import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("/home/abhinav/Documents/Machine Learning A-Z (Model Selection)/Classification/Data.csv")
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
print(X)
print(Y)
# training and testing the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
# feature scalling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.metrics import accuracy_score
# logistic
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=0)
lr.fit(X_train,Y_train)
lr_accuracy=accuracy_score(Y_test,lr.predict(X_test))
# knn
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=10,metric="minkowski",p=2)
knn.fit(X_train,Y_train)
knn_accuracy=accuracy_score(Y_test,knn.predict(X_test))
# svm linear
from sklearn.svm import SVC
svm_linear=SVC(kernel="linear",random_state=0)
svm_linear.fit(X_train,Y_train)
svm_linear_accuracy=accuracy_score(Y_test,svm_linear.predict(X_test))
# svm kernel
from sklearn.svm import SVC
svm_kernel=SVC(kernel="rbf",random_state=0)
svm_kernel.fit(X_train,Y_train)
svm_kernel_accuracy=accuracy_score(Y_test,svm_kernel.predict(X_test))
# naive bayes
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X_train,Y_train)
nb_accuracy=accuracy_score(Y_test,nb.predict(X_test))
# decision tree
from sklearn.tree import DecisionTreeClassifier
dct=DecisionTreeClassifier(criterion="entropy",random_state=0)
dct.fit(X_train,Y_train)
dct_accuracy=accuracy_score(Y_test,dct.predict(X_test))
# random forest
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=10,criterion="entropy",random_state=0)
rf.fit(X_train,Y_train)
rf_accuracy=accuracy_score(Y_test,rf.predict(X_test))
print(f"Logistic:{lr_accuracy}")
print(f"knn:{knn_accuracy}")
print(f"svm linear:{svm_linear_accuracy}")
print(f"svm kernel:{svm_kernel_accuracy}")
print(f"decision:{dct_accuracy}")
print(f"random:{rf_accuracy}")
print(f"naive bayes:{nb_accuracy}")