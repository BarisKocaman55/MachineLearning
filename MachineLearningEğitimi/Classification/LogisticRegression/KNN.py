import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#------------------Dataları Çekme-----------------
data= pd.read_csv('veriler.csv')
x_values = data.iloc[0:, 1:4].values
y_values = data.iloc[0:, -1:].values

#------------------Train ve Test------------------
x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size = 0.33, random_state = 0)

#------------------Verilerin Ölçeklenmesi---------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


#-----------------LogisticRegreesion-------------
from sklearn.linear_model import LogisticRegression
logistic_reg = LogisticRegression(random_state = 0)
logistic_reg.fit(x_train, y_train)

y_predict = logistic_reg.predict(x_test)
print("---------Predictions--------")
print(y_predict)
print("-------Actual Values--------")
print(y_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
print("------FOR LOGISTIC------")
print(cm)

#---------------------KNN ALGORITHM----------------------
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1, metric = 'minkowski')
knn.fit(x_train, y_train)
y_predict1 = knn.predict(x_test)

cm1 = confusion_matrix(y_test, y_predict1)
print("------FOR KNN-------")
print(cm1)




