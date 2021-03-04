import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_excel("/home/abhinav/PycharmProjects/Machine Learning Udemy/ANN_Project_Electricity/Folds5x2_pp.xlsx")
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from tensorflow.keras.models import load_model
model=load_model('model.h5')
y_pred=model.predict(X_test)
np.set_printoptions(precision=2)
y_pred=y_pred.flatten()
Y_test=Y_test.flatten()
print(y_pred)
print(Y_test)
print(model.predict(sc.transform([[8.34,40.77,1010.84,90.01]])))
print(np.concatenate((y_pred.reshape(len(y_pred),1), Y_test.reshape(len(Y_test),1)),1))
plt.plot(range(1,len(Y_test)+1),Y_test,color="red")
plt.plot(range(1,len(Y_test)+1),y_pred,color="blue")
plt.xlim([0,50])
plt.show()