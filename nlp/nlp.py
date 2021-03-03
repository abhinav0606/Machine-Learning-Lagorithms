import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords
import re
dataset=pd.read_csv("/home/abhinav/Documents/Machine Learning/Machine Learning A-Z (Codes and Datasets)/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing/Python/Restaurant_Reviews.tsv",delimiter="\t",quoting=3)
corpus=[]
for i in range(1000):
    review=re.sub("[^a-zA-Z]"," ",dataset["Review"][i])
    review=review.lower()
    review=review.split()
    swords=stopwords.words("english")
    swords.remove("not")
    s=PorterStemmer()
    review=[s.stem(i) for i in review if i not in swords]
    review=" ".join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
Y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.naive_bayes import GaussianNB
c=GaussianNB()
c.fit(X_train,Y_train)

y_pred=c.predict(X_test)

y_pred=np.concatenate((y_pred.reshape(len(y_pred),1),Y_test.reshape(len(Y_test),1)),1)
print(y_pred)

f=[]
new_review=input("Enter the review")
new_review=re.sub("[^a-zA-Z]"," ",new_review)
new_review=new_review.lower()
new_review=new_review.split()
stop=stopwords.words("english")
stop.remove("not")
ps=PorterStemmer()
new_review=[ps.stem(i) for i in new_review if i not in stop]
new_review=" ".join(new_review)
f.append(new_review)

new_x=cv.transform(f).toarray()
prediction=c.predict(new_x)
if prediction[0]==1:
    print("Happy")
else:
    print("Not happy")