import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("/home/abhinav/Documents/Final Folder/Dataset/breast_cancer.csv")
print(dataset)
X=dataset.iloc[:,1:-1].values
Y=dataset.iloc[:,-1].values
print(X)
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=0)
lr.fit(X_train,Y_train)
y_pred=lr.predict(X_test)
z=np.concatenate((y_pred.reshape(len(y_pred),1),Y_test.reshape(len(Y_test),1)),1)
print(z)
from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_pred,Y_test))
print(accuracy_score(y_pred,Y_test))
from sklearn.model_selection import cross_val_score
accuracy=cross_val_score(estimator=lr,X=X_train,y=Y_train,cv=10)
print(accuracy.mean()*100)
print(accuracy.std()*100)
