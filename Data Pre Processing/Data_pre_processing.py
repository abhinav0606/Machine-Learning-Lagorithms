import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv("/home/abhinav/Documents/Machine Learning/Machine Learning A-Z (Codes and Datasets)/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Python/Data.csv")
print(dataset)
# as from here we will have the dataset and we want that we will predict the future
# of the company based on this dataset so there is a dependent variable which will help
# us to do that
# independent variable
X=dataset.iloc[:,:-1].values
# dependent variable
Y=dataset.iloc[:,-1].values
print(X)
print(Y)
# taking care of the missing values
# now there are two ways to do that
# one is if we have large database then we can ignore this values
# other one is we can fill this value by taking mean and only interger value will be filler
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy="mean")
imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])
print(X)

# encoding the data category wise

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
ct=ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[0])],remainder="passthrough")
X=np.array(ct.fit_transform(X))
print(X)

# encoding the dependent variable
lb=LabelEncoder()
Y=lb.fit_transform(Y)
print(Y)

# training the dataset and testing them also
# test_size means how many have to go in to the testing
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)


# feature scalling
# this we will apply to get the access the values in a range

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train[:,3:]=sc.fit_transform(X_train[:,3:])
X_test[:,3:]=sc.transform(X_test[:,3:])
print(X_train)
print(X_test)