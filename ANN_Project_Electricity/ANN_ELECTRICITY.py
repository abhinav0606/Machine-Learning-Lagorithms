# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

# importing the dataset

dataset=pd.read_excel("/home/abhinav/PycharmProjects/Machine Learning Udemy/ANN_Project_Electricity/Folds5x2_pp.xlsx")
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
print(X)
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
print(Y_test)
# feature scalling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
print(X_train)
print(X_test)
# for regression point of view always use mse (mean_squared_error)
# building the ann
ann=tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
# as we will not use probability so we will not use sigmoid as activation in the third layer
ann.add(tf.keras.layers.Dense(units=1))
ann.compile(optimizer='adam',loss="mean_squared_error",metrics=['accuracy'])
ann.fit(X_train,Y_train,batch_size=32,epochs=100)
y_pred=ann.predict(X_test)
ann.save('model.h5')