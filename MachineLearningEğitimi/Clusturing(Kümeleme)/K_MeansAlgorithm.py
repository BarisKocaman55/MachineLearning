import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('musteriler.csv')
x = data.iloc[:,2:4].values
y = data.iloc[:,4:5].values


#------------K_MEANS ALGORITHM---------------
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, init = 'k-means++')
kmeans.fit(x)

print("Cluster Centers")
print(kmeans.cluster_centers_)


#-----------DECIDE K VALUE------------
results = []
for i in range(1, 11):
    kmeans1 = KMeans(n_clusters = i, init = 'k-means++', random_state = 123)
    kmeans1.fit(x)
    results.append(kmeans.inertia_)
    
    
plt.plot(range(1,11), results)
plt.show()    