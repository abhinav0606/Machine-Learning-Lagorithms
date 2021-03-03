import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# datapreprocessing

dataset=pd.read_csv("/home/abhinav/Documents/Machine Learning/Machine Learning A-Z (Codes and Datasets)/Part 6 - Reinforcement Learning/Section 33 - Thompson Sampling/Python/Ads_CTR_Optimisation.csv")



# Thompson sampling
d=10
N=10000
ads_selected=[]
ni_1=[0]*d
ni_0=[0]*d
total_reward=0
for n in range(N):
    ad=0
    max_bound=0
    for i in range(d):
        theta=random.betavariate(ni_1[i]+1,ni_0[i]+1)
        if theta>max_bound:
            max_bound=theta
            ad=i
    ads_selected.append(ad)
    reward=dataset.values[n,ad]
    if reward==1:
        ni_1[ad]=ni_1[ad]+1
    else:
        ni_0[ad]=ni_0[ad]+1
    total_reward=total_reward+reward


dicty={}
for i in ads_selected:
    dicty[i]=ads_selected.count(i)
plt.bar(dicty.keys(),dicty.values())
plt.show()