import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# data preprocessing
dataset=pd.read_csv("/home/abhinav/Documents/Machine Learning/Machine Learning A-Z (Codes and Datasets)/Part 3 - Classification/Section 15 - K-Nearest Neighbors (K-NN)/Python/Social_Network_Ads.csv")
X=dataset.iloc[:,:2].values
Y=dataset.iloc[:,-1].values

# splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
# feature scalling bcz this will improve the predictability
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
print(X_train)
print(X_test)


# now training the dataset on the logistic regression class
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=0)
lr.fit(X_train,Y_train)
# predicting the new result
print(lr.predict(sc.transform([[50,89000]])))

# predicting the test set results
y_pred=lr.predict(X_test)
z=np.concatenate((y_pred.reshape(len(y_pred),1),Y_test.reshape(len(Y_test),1)),1)
print(z)

# confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix
print(confusion_matrix(Y_test,y_pred))
print(accuracy_score(Y_test,y_pred))

# visualising the data

for i in sc.inverse_transform(X_train):
    if lr.predict(sc.transform([[i[0],i[1]]]))==0:
        plt.scatter(i[0],i[1],color="red")
    else:
        plt.scatter(i[0],i[1],color="blue")
u=sc.inverse_transform(X_train)
for i in range(len(u)):
    if Y_train[i]==0:
        plt.scatter(u[i][0],u[i][1],color="red")
    else:
        plt.scatter(u[i][0],u[i][1],color="blue")
plt.show()