import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("/home/abhinav/Documents/Machine Learning/Machine Learning A-Z (Codes and Datasets)/Part 4 - Clustering/Section 25 - Hierarchical Clustering/Python/Mall_Customers.csv")
X=dataset.iloc[:,[3,4]].values
print(X)
# create a dendrogram
import scipy.cluster.hierarchy as sh
dendrogram=sh.dendrogram(sh.linkage(X,method="ward"))
plt.title("Dendrogram")
plt.xlabel("Clusters")
plt.ylabel("Distance")
plt.show()

# training the dataset
from sklearn.cluster import AgglomerativeClustering
ag=AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")
result=ag.fit_predict(X)

plt.scatter(X[result==0,0],X[result==0,1],c="red",label="cluster1")
plt.scatter(X[result==1,0],X[result==1,1],c="blue",label="cluster2")
plt.scatter(X[result==2,0],X[result==2,1],c="green",label="cluster3")
plt.scatter(X[result==3,0],X[result==3,1],c="yellow",label="cluster4")
plt.scatter(X[result==4,0],X[result==4,1],c="magenta",label="cluster5")
plt.title("Main Plotting")
plt.xlabel("Income")
plt.ylabel("Spending Hours")
plt.legend()
plt.show()