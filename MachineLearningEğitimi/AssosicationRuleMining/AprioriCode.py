import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('sepet.csv', header = None)

t = []
for i in range(1, 7501):
    t.append([str(data.values[i,j]) for j in range (0,20)])


 
from apyori import apriori
rules = apriori(t, min_support = 0.01, min_confidence = 0.2, min_lift = 3, min_length = 2)


print(list(rules))






