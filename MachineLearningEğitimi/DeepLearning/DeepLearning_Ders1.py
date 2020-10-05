import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from tensorflow import keras
#from tensorflow.keras import layers


#-------------DATA PROCESSING-------------
data = pd.read_csv('Churn_Modelling.csv')
x = data.iloc[:,3:13].values
y = data.iloc[:,13:14].values

from sklearn import preprocessing 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
le = preprocessing.LabelEncoder()
x[:,1:2] = le.fit_transform(x[:,1:2]).reshape(-1, 1) #Country Encoding
x[:,2:3] = le.fit_transform(x[:,2:3]).reshape(-1, 1) #Gender Binarize

ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float), [1])], remainder = "passthrough")
x = ohe.fit_transform(x)
x = x[:, 1:]


print(x[:, 1:2])



#------------------TRAIN/TEST----------------------
from sklearn.preprocessing import StandardScaler
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()







