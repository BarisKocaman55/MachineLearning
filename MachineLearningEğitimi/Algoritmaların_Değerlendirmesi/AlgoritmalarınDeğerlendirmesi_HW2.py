import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures


data = pd.read_csv('maaslar_yeni.csv')
x_values = data.iloc[:, 2:3].values
wage_values = data.iloc[:, 5:6].values


#--------------Linear Regression-----------------
from sklearn.linear_model import LinearRegression

x_train1, x_test1, y_train1, y_test1 = train_test_split(x_values, wage_values, test_size = 0.2, random_state = 0)
linear_reg = LinearRegression()
linear_reg.fit(x_train1, y_train1)
wage_predict = linear_reg.predict(x_test1)

model = sm.OLS(linear_reg.predict(x_values) ,x_values)
print(model.fit().summary())



#--------------Polynomial Regression----------------
linear_regression1 = LinearRegression()
polynomial_regression = PolynomialFeatures(degree = 2)
x_poly = polynomial_regression.fit_transform(x_values)
linear_regression1.fit(x_poly[:, 2:3], wage_result)

