import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

#--------------READING DATA----------------
data = pd.read_csv('odev_tenis.csv')
outlook = data.iloc[:,0:1].values
temperature = data.iloc[:,1:2].values
windy = data.iloc[:,3:4].values
play = data.iloc[:,-1:].values
humudity = data.iloc[:,2:3].values


#--------------PREPARING DATA--------------
le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()

outlook[:,0] = le.fit_transform(data.iloc[:,0])
outlook = ohe.fit_transform(outlook).toarray()

windy[:,0] = le.fit_transform(data.iloc[:,0])
windy = ohe.fit_transform(windy).toarray()

play[:,0] = le.fit_transform(play[:,0])

'''
print("----------OUTLOOK----------")
print(outlook)
print("---------WINDY-------------")
print(windy)
print("-------------PLAY-----------")
print(play)
'''

#--------------DATAFRAME OLUSTURMA--------------
outlook_result = pd.DataFrame(data=outlook, index=range(14), columns=['sunny', 'overcast', 'rainy'])
windy_result = pd.DataFrame(data=windy[:,0:1], index=range(14), columns=['windy'])
temperature_result = pd.DataFrame(data=temperature, index=range(14), columns=['temperature'])
play_result = pd.DataFrame(play[:,0:1], index=range(14), columns=['Gender'])
humudity_result = pd.DataFrame(data=humudity, index=range(14), columns=['humudity'])
'''
print(outlook_result)
print(windy_result)
print(temperature_result)
print(play_result)
'''

#---------------DATAFRAME LERİ BİRLEŞTİRME---------------
result1 = pd.concat([outlook_result, temperature_result], axis=1)
result2 = pd.concat([windy_result, play_result], axis=1)
final_result = pd.concat([result1, result2], axis=1)


#---------------TRAIN-TEST VERİLERİNİN OLUŞTURULMASI---------------
x_train, x_test, y_train, y_test = train_test_split(final_result, humudity_result, test_size=0.2, random_state=0)


#-----------------MULTIPLE LİNEAR REGRESSION------------------
linear_regression = LinearRegression()
linear_regression.fit(x_train, y_train)

humadity_predict = linear_regression.predict(x_test)


#---------------BACKWARD ELEMINATION----------------
import statsmodels.api as sm
X = np.append(arr=np.ones((14,1)).astype(int), values=final_result, axis=1)

print("----------First Try---------")
x_l = final_result.iloc[:,[0,1,2,3,4,5]].values
x_l = np.array(x_l, dtype=float)
model = sm.OLS(humudity,x_l).fit()
print(model.summary())

print("----------Second Try---------")
x_l = final_result.iloc[:,[0,1,3,4,5]].values
x_l = np.array(x_l, dtype=float)
model = sm.OLS(humudity,x_l).fit()
print(model.summary())

print("----------Third Try---------")
x_l = final_result.iloc[:,[0,1,3,5]].values
x_l = np.array(x_l, dtype=float)
model = sm.OLS(humudity,x_l).fit()
print(model.summary())


#------------------TRAINING ACCORDING TO BACKWARD ELEMINATION------------------
final_result1 = final_result.iloc[:,0:2]
final_result2 = final_result.iloc[:,3:4]
final_result3 = final_result.iloc[:,-1]

last_result1 = pd.concat([final_result1, final_result2], axis=1)
last_result2 = pd.concat([last_result1, final_result3], axis=1)

x_train2, x_test2, y_train2, y_test2 = train_test_split(last_result2, humudity, test_size=0.2, random_state=0)
linear_regression.fit(x_train2, y_train2)

humudity_predict2 = linear_regression.predict(x_test2)
print(data)