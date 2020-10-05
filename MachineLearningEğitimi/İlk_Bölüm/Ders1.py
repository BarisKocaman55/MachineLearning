import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('veriler.csv')
print(data.head())

new_data = data[['boy', 'kilo']]
print(new_data)

class deneme:
    boy = 180
    def kos(self, a):
        sonuc = a + 10
        return sonuc
    
deneme = deneme()
print(deneme.boy)
print(deneme.kos(90))