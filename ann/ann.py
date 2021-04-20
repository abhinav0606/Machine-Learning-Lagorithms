# https://freefrontend.com/bootstrap-payment-forms/
import pandas as pd
import numpy as np
import tensorflow as tf
dataset=pd.read_csv("/home/abhinav/Documents/Machine Learning/Machine Learning A-Z (Codes and Datasets)/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)/Python/Churn_Modelling.csv")
X=dataset.iloc[:,3:-1].values
Y=dataset.iloc[:,-1].values
# label encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X[:,2]=le.fit_transform(X[:,2])
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X=np.array(ct.fit_transform(X))
# splitting
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
# create ann
ann=tf.keras.models.Sequential()
# adding first layer
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
# second layer
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
# final output
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
ann.fit(X_train,Y_train,batch_size=32,epochs=100)
print((ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))>0.5)