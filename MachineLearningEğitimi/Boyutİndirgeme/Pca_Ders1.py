import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#--------------READ DATA------------
data = pd.read_csv('Wine.csv')
x = data.iloc[:,0:13].values
y = data.iloc[:,13:14].values


#--------------TRAIN-TEST-------------
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


#--------------STANDARTIZE-------------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train2 = sc.fit_transform(x_train)
x_test2 = sc.fit_transform(x_test)


from sklearn.decomposition import PCA
pca = PCA(n_components = 2)

x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)


from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(random_state = 0)
classifier2 = LogisticRegression(random_state = 0)

classifier1.fit(x_train, y_train)
classifier2.fit(x_train2, y_train)


from sklearn.metrics import confusion_matrix
print("gerck / pcasiz")
y_pred1 = classifier1.predict(x_test)
cm1 = confusion_matrix(y_test, y_pred1)
print(cm1)

print("gercek / pca ile")
y_pred2 = classifier2.predict(x_test2)
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)

print("pcasiz ve pcali")
cm3 = confusion_matrix(y_pred1, y_pred2)
print(cm3)



#-------------------LDA ALGORITHM-------------------
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)

x_train_lda = lda.fit_transform(x_train, y_train)
x_test_lda = lda.transform(x_test)

classifier_lda = LogisticRegression(random_state = 0)
classifier_lda.fit(x_train_lda, y_train)

y_pred_lda = classifier_lda.predict(x_test_lda)
cm_lda = confusion_matrix(y_test, y_pred_lda)
print("-------AFTER LDA-------")
print(cm_lda)






