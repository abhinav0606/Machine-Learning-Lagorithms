import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# datapreprocessing
dataset=pd.read_csv("/home/abhinav/Documents/Machine Learning/Machine Learning A-Z (Codes and Datasets)/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Python/Ads_CTR_Optimisation.csv")
# analyzing the maximum ucb
import math
N=10000
d=10
number_of_selected_ads=[0]*d
sum_of_reward=[0]*d
total_reward=0
ads_selected=[]
# creating and analyzing the maximum ucbs
for n in range(N):
    ad=0
    max_upper_bound=0
    for i in range(d):
        if number_of_selected_ads[i]>0:
            avg_reward=sum_of_reward[i]/number_of_selected_ads[i]
            delta=math.sqrt((1.5*math.log(n+1))/number_of_selected_ads[i])
            ucb=avg_reward+delta
        else:
            ucb=1e400
        if ucb>max_upper_bound:
            max_upper_bound=ucb
            ad=i
    ads_selected.append(ad)
    number_of_selected_ads[ad]=number_of_selected_ads[ad]+1
    reward=dataset.values[n,ad]
    sum_of_reward[ad]=sum_of_reward[ad]+reward
    total_reward=total_reward+reward
d={}
for i in ads_selected:
    d[i]=ads_selected.count(i)
print(d)
plt.bar(d.keys(),d.values())
plt.xlabel("Ad number")
plt.ylabel("Number of clicks")
plt.title("Checking the best ads")
plt.show()