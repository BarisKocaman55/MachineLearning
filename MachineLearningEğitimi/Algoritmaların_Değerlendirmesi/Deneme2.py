import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import r2_score

#----------------TAKEING DATA FROM FILE-------------------
data = pd.read_csv('maaslar_yeni.csv')
print(data.corr())

x_values = data.iloc[:, 2:5].values
wage_values = data.iloc[:, 5:].values

x_values1 = data.iloc[:, 2:3].values
x_values2 = data.iloc[:, 4:5].values

x_result1 = pd.DataFrame(data = x_values1, index=range(30), columns = ['Unvan Seviyesi'])
x_result2 = pd.DataFrame(data = x_values2, index=range(30), columns = ['Puan'])
finalx_result = pd.concat([x_result1, x_result2], axis=1)

#----------------Multıple LINEAR REGRESSION-----------------------
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(finalx_result, wage_values)

#print("--------MULTIPLE LINEAR REGRESSION------------")
#model1 = sm.OLS(linear_reg.predict(x_values) ,x_values)
#print(model1.fit().summary())

print("--------MULTIPLE LINEAR REGRESSION NEW------------")
model3 = sm.OLS(linear_reg.predict(finalx_result) ,finalx_result)
print(model3.fit().summary())
#
print("------------Multiple Linear Regression R2 Value-------------------")
print(r2_score(wage_values, linear_reg.predict(finalx_result)))

#----------------POLYNOMIAL REGRESSION----------------------------
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x_values)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, wage_values)

print("--------POLYNOMIAL REGRESSION------------")
model2=sm.OLS(lin_reg2.predict(poly_reg.fit_transform(x_values)),x_values)
print(model2.fit().summary())

print('-----------------Polynomial R2 Value------------------')
print(r2_score(wage_values, lin_reg2.predict(poly_reg.fit_transform(x_values))))



#----------------------------DECISION TREE REGRESSION--------------------------
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state = 0)
r_dt.fit(x_values, wage_values)

print('---------------Decision Tree OLS-----------------')
model4=sm.OLS(r_dt.predict(x_values),x_values)
print(model4.fit().summary())

print('----------------Decision Tree R2 Value---------------')
print(r2_score(wage_values, r_dt.predict(x_values)))



#----------------------------RANDOM FOREST REGRESSION----------------
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10, random_state = 0)
rf_reg.fit(x_values, wage_values.ravel())

print('Random Forest OLS')
model5=sm.OLS(rf_reg.predict(x_values),x_values)
print(model5.fit().summary())



print('Random Forest R2 degeri')
print(r2_score(wage_values, rf_reg.predict(x_values)))
