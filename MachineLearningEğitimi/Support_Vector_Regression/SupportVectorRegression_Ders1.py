import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('maaslar.csv')
x = data.iloc[:,1:2].values
y = data.iloc[:,2:].values
x1 = data.iloc[:,1:2]
y1 = data.iloc[:,2:] 

#-----------------------------------WITH LINEAR REGRESSION------------------------------
linear_regression = LinearRegression()
linear_regression.fit(x, y)

#------VIRTUALIZATION--------
plt.scatter(x, y, color='red')
plt.plot(x, linear_regression.predict(x), color='blue')
plt.show()


#---------------------------------WITH POLYNOMIAL REGRESSION-----------------------------
linear_regression1 = LinearRegression()
polynomial_regression = PolynomialFeatures(degree = 2)
x_poly = polynomial_regression.fit_transform(x)
linear_regression1.fit(x_poly, y)
plt.scatter(x, y, color='red')
plt.plot(x, linear_regression1.predict(polynomial_regression.fit_transform(x)))
plt.show()


linear_regression1 = LinearRegression()
polynomial_regression = PolynomialFeatures(degree = 4)
x_poly = polynomial_regression.fit_transform(x)
linear_regression1.fit(x_poly, y)
plt.scatter(x, y, color='green')
plt.plot(x, linear_regression1.predict(polynomial_regression.fit_transform(x)))
plt.show()



from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
sc2 = StandardScaler()

x_olcekli  = sc1.fit_transform(x)
y_olcekli = np.ravel(sc2.fit_transform(y.reshape(-1,1)))


from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli, y_olcekli)

plt.scatter(x_olcekli, y_olcekli)
plt.plot(x_olcekli, svr_reg.predict(x_olcekli), color='orange')
plt.show()






