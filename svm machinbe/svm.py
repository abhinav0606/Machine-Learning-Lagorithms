import numpy as np
import pandas as pd
import matplotlib.pyplot as ptl
dataset=pd.read_csv("/home/abhinav/Documents/Machine Learning/Machine Learning A-Z (Codes and Datasets)/Part 3 - Classification/Section 16 - Support Vector Machine (SVM)/Python/Social_Network_Ads.csv")
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.svm import SVC
svc=SVC(kernel="linear",random_state=0)
svc.fit(X_train,Y_train)
print(svc.predict(X_test))
y_pred=svc.predict(X_test)
z=np.concatenate((y_pred.reshape(len(y_pred),1),Y_test.reshape(len(Y_test),1)),1)
print(z)
from sklearn.metrics import accuracy_score,confusion_matrix
print(confusion_matrix(Y_test,y_pred))
print(accuracy_score(Y_test,y_pred))