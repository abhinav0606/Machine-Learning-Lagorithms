import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
data=pd.read_csv("/home/abhinav/Documents/Machine Learning/Machine Learning A-Z (Codes and Datasets)/Part 4 - Clustering/Section 24 - K-Means Clustering/Python/Mall_Customers.csv")
X=data.iloc[:,[3,4]].values

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.xlabel("Number of Cluster")
plt.ylabel("WCSS")
plt.show()

kmeans=KMeans(n_clusters=5,init="k-means++",random_state=42)
ykmeans=kmeans.fit_predict(X)

plt.scatter(X[ykmeans==0,0],X[ykmeans==0,1],c="red",label="c1")
plt.scatter(X[ykmeans==1,0],X[ykmeans==1,1],c="blue",label="c1")
plt.scatter(X[ykmeans==2,0],X[ykmeans==2,1],c="green",label="c1")
plt.scatter(X[ykmeans==3,0],X[ykmeans==3,1],c="yellow",label="c1")
plt.scatter(X[ykmeans==4,0],X[ykmeans==4,1],c="magenta",label="c1")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c="cyan",label="Centroid")
plt.xlabel("Anual income")
plt.ylabel("Spending percentage")
plt.legend()
plt.show()