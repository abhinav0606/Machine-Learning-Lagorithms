from apyori import apriori
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# dataprocessing
dataset=pd.read_csv("/home/abhinav/Documents/Machine Learning/Machine Learning A-Z (Codes and Datasets)/Part 5 - Association Rule Learning/Section 29 - Eclat/Python/Market_Basket_Optimisation.csv",header=None)
t=[]
for i in range(7501):
    t.append([str(dataset[j][i]) for j in range(20)])
rules=apriori(transactions=t,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2,max_length=2)
result=list(rules)
def get_relation(result):
    lhs=[tuple(i[2][0][0])[0] for i in result]
    rhs=[tuple(i[2][0][1])[0] for i in result]
    support=[i[1] for i in result]
    return list(zip(lhs,rhs,support))

new_df=pd.DataFrame(get_relation(result),columns=["Left Hand Side","Right Hand Side","Support"])
print(new_df)

sorted_df=new_df.nlargest(n=10,columns=["Support"])
print("--------------------------------------------------")
print(sorted_df)