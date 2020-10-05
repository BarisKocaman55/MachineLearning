import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('veriler.csv')
print(data)

countries = data.iloc[:,0:1].values
boy_kilo_yas = data.iloc[:,1:4].values
cinsiyet = data.iloc[:,-1].values


print("-----Before Transform-----")
print(countries)

le = preprocessing.LabelEncoder()
countries[:,0] = le.fit_transform(data.iloc[:,0])

print("-----After Transform-----")
print(countries)

print("-----After Transform it to Array-----")
ohe = preprocessing.OneHotEncoder()
countries = ohe.fit_transform(countries).toarray()
print(countries)



#------------Dataları DataFrame içerisine Atmak---------
country_result = pd.DataFrame(data=countries, index=range(22), columns=['fr','tr','us'])
age_result = pd.DataFrame(data=boy_kilo_yas, index=range(22), columns=['boy','kilo','yas'])
cinsiyet_result = pd.DataFrame(data=cinsiyet, index=range(22), columns=['cinsiyet'])


print("------Result-----")
print(country_result)
print(age_result)
print(cinsiyet_result)


#------DataFrameleri Birleştirme------
final_result1 = pd.concat([country_result, age_result], axis=1)
print(final_result1)

print("---------------------------")
final_result2 = pd.concat([final_result1, cinsiyet_result], axis=1)
print(final_result2)



#------------DATALARI TEST VE TRAIN OLARAK BOLME-----------------
x_train, x_test, y_train, y_test = train_test_split(final_result1, cinsiyet_result, test_size=0.33, random_state=0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)






