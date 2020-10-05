import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


data = pd.read_csv('satislar.csv')

months = data[['Aylar']]
sales = data[['Satislar']]

print(sales)


x_train, x_test, y_train, y_test = train_test_split(months, sales, test_size=0.33, random_state=0)

'''
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
y_train = sc.fit_transform(y_train)
y_test = sc.fit_transform(y_test)
'''

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

tahmin = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.title("Sales")
plt.xlabel("Months")
plt.ylabel("Sales")
plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))


