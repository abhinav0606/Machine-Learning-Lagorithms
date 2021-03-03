# importing the modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# datapreprocessing
dataset=pd.read_csv("/home/abhinav/Documents/Machine Learning/Machine Learning A-Z (Codes and Datasets)/Part 5 - Association Rule Learning/Section 28 - Apriori/Python/Market_Basket_Optimisation.csv",header=None)
print(dataset)
# now installing the apyori module that will take the format of list as the input
t=[]
for i in range(7501):
    t.append([str(dataset[j][i]) for j in range(20)])
# print(t)
# training the apriori algorithm
from apyori import apriori
rules=apriori(transactions=t,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2,max_length=2)
result=list(rules)
print(result)

# inspecting which item is matching with which item
def relation(result):
    lhs=[tuple(i[2][0][0])[0] for i in result]
    rhs=[tuple(i[2][0][1])[0] for i in result]
    support=[i[1] for i in result]
    confidence=[i[2][0][2] for i in result]
    Lift=[i[2][0][3] for i in result]
    return list(zip(lhs,rhs,support,confidence,Lift))
resulting_df=pd.DataFrame(relation(result),columns=["LHS","RHS","Support","Confidence","Lift"])
print(resulting_df)
print("---------------------------------------------------")
# sorting
sorted=resulting_df.nlargest(n=10,columns=["Lift"])
print(sorted)