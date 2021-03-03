import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("/home/abhinav/Documents/Machine Learning/Machine Learning A-Z (Codes and Datasets)/Part 9 - Dimensionality Reduction/Section 43 - Principal Component Analysis (PCA)/Python/Wine.csv")
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=0)
lr.fit(X_train,Y_train)

from sklearn.metrics import confusion_matrix,accuracy_score
y_pred=lr.predict(X_test)
print(confusion_matrix(Y_test,y_pred))
print(accuracy_score(Y_test,y_pred))