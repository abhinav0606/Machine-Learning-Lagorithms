import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("/home/abhinav/Documents/Machine Learning/Machine Learning A-Z (Codes and Datasets)/Part 3 - Classification/Section 17 - Kernel SVM/Python/Social_Network_Ads.csv")
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
print(X)
print(Y)

# datapreprocessing part
# training and testing and splitting
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

# feature scalling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
print(X_train)
print(X_test)

# applying the classification class
from sklearn.svm import SVC
svc=SVC(kernel="rbf",random_state=0)
svc.fit(X_train,Y_train)
print(svc.predict(sc.transform([[30,60000]])))
y_pred=svc.predict(X_test)
z=np.concatenate((y_pred.reshape(len(y_pred),1),Y_test.reshape(len(Y_test),1)),1)
print(z)
from sklearn.metrics import accuracy_score,confusion_matrix
print(confusion_matrix(Y_test,y_pred))
print(accuracy_score(Y_test,y_pred))