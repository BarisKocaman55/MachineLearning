import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle 

#-------------READING DATA---------------
url = "https://bilkav.com/satislar.csv"
data = pd.read_csv(url)
x = data.iloc[:,0:1].values
y = data.iloc[:, 1:2].values


#------------TRAIN AND TEST--------------
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


#------------LINEAR REGRESSION-----------
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train)

y_pred = linear_reg.predict(x_test)


#------------ACCURACY SCORE--------------
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
print("Accuracy = " , score)


#-----------SAVE THE MODEL--------------
file = "file.kayit"
pickle.dump(linear_reg, open(file, 'wb'))

loaded = pickle.load(open(file, 'rb'))
print("Test Values")
print(loaded.predict(x_test))