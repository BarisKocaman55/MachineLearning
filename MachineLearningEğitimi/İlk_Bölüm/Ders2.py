import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

data = pd.read_csv('eksikveriler.csv')
print(data)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

print("-----Before Transform-----")
yas = data.iloc[:,1:4].values
print(yas)
imputer = imputer.fit(yas[:,1:4])
yas[:,1:4] = imputer.transform(yas[:,1:4])
print("-----After Transform-----")
print(yas)