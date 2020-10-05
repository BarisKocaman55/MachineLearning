import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.impute import SimpleImputer


#--------------Dataları Çekme------------------
data = pd.read_csv('veriler.csv')
countries = data.iloc[:,0:1].values
height_weight_age = data.iloc[:,1:4].values
gender = data.iloc[:,-1:].values

#-----------Ülkeleri ve Cinsiyeti Düzenleme---------
le = preprocessing.LabelEncoder()
countries[:,0] = le.fit_transform(data.iloc[:,0])
print("-----Before Transformation-----")
print(countries)

print("-----After Transformation-----")
ohe = preprocessing.OneHotEncoder()
countries = ohe.fit_transform(countries).toarray()
print(countries)

le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()
gender[:,-1] = le.fit_transform(data.iloc[:,-1])
gender = ohe.fit_transform(gender).toarray()
print(gender)

#---------------DataFrame Oluşturma---------------
country_result = pd.DataFrame(data=countries, index=range(22), columns=['fr', 'tr', 'us'])
height_weight_age_result = pd.DataFrame(data=height_weight_age, index=range(22), columns=['boy', 'kilo', 'yas'])
gender_result = pd.DataFrame(data=gender[:,0:1], index=range(22), columns=['cinsiyet'])

#------------------DataFrameleri Birleştirme------------------
result = pd.concat([country_result, height_weight_age_result], axis=1)
final_result = pd.concat([result, gender_result], axis=1)
print(final_result)


#---------------Test-Train------------------
x_train, x_test, y_train, y_test = train_test_split(result, gender_result, test_size=0.33, random_state=0)



#-------------Multiple Linear Regression-----------------
linear = LinearRegression()
linear.fit(x_train, y_train)

y_pred = linear.predict(x_test)


#-----------------Boy Tahmini-------------------
height = final_result.iloc[:,3:4].values
left_values = final_result.iloc[:,:3]
right_values = final_result.iloc[:,4:]

learning_model = pd.concat([left_values, right_values], axis=1)

x_train1, x_test1, y_train1, y_test1 = train_test_split(learning_model, height, test_size=0.2, random_state=0)
height_linear = LinearRegression()
height_linear.fit(x_train1, y_train1)
height_prediction = height_linear.predict(x_test1)


import statsmodels.api as sm
X = np.append(arr = np.ones((22,1)).astype(int), values=learning_model, axis=1)

x_l = learning_model.iloc[:,[0,1,2,3,4,5]].values
x_l = np.array(x_l, dtype=float)
model = sm.OLS(height,x_l).fit()
print("----------MODEL1 BACKWARD ELEMINATION----------")
print(model.summary())


x_l = learning_model.iloc[:,[0,1,2,3,5]].values
x_l = np.array(x_l, dtype=float)
model = sm.OLS(height,x_l).fit()
print("----------MODEL2 BACKWARD ELEMINATION----------")
print(model.summary())





















