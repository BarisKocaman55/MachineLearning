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
    

#----------------HIERARCHICAL CLUSTURING------------------
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')
y_pred = ac.fit_predict(x)
print(y_pred)


plt.scatter(x[y_pred == 0,0], x[y_pred == 0,1], s = 100, color = 'red')
plt.scatter(x[y_pred == 1,0], x[y_pred == 1,1], s = 100, color = 'green')
plt.scatter(x[y_pred == 2,0], x[y_pred == 2,1], s = 100, color = 'blue')
plt.scatter(x[y_pred == 3,0], x[y_pred == 3,1], s = 100, color = 'yellow')

plt.show()


#Dendrogram 
import scipy.cluster.hierarchy as sch
#dendogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
#plt1.show()


