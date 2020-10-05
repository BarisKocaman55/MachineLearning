import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

data = pd.read_csv('Ads_CTR_Optimisation.csv')

#-----------------------RANDOM SELECTION-------------------------
#Generation random numbers
N = 10000
d = 10
summary = 0
selected = []
for n in range(0, N):
    ad = random.randrange(d)
    selected.append(ad)
    reward = data.values[n, ad] #verilerdeki n. satırdaki ad 1 ise ödül 1 artıyor, değilse 0 artıyor
    summary = summary + reward


#plt.hist(selected)
#plt.show()


#----------------------UPPER CONFIDENCE BOUNDE (UCB)-----------------------
import math
N1 = 10000
d1 = 10
reward = [0] * d1
sum_reward = 0
clicked = [0] * d1
selected1 = []
summ = 0

for n in range(0, N1):
    ad = 0
    max_ucb = 0
    for i in range(0, d1):
        if (clicked[i] > 0):
            avarage = reward[i] / clicked[i]
            delta = math.sqrt(3/2 * math.log(n) / clicked[i])
            ucb = avarage + delta
        else:
            ucb = N * 10
        if max_ucb < ucb:
            max_ucb = ucb
            ad = i
    
    selected1.append(ad)
    clicked[ad] += 1
    reward1 = data.values[n, ad]
    reward[ad] = reward[ad] + reward1
    summ = summ + reward1
    
    
print("-------REWARD--------")
print(summ)
import matplotlib.pyplot as plt
plt.hist(selected1)
plt.show()