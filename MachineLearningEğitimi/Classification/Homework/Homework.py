import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#---------------READING DATA FROM EXCEL--------------
data = pd.read_excel('Iris.xls')
x = data.iloc[:, 0:4].values
y = data.iloc[:, 4:].values


#---------------PRAPARING TRAIN AND TEST DATAS-------
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


#---------------LOGISTIC REGRESSION------------------
from sklearn.linear_model import LogisticRegression
logistic_reg = LogisticRegression(random_state = 0)
logistic_reg.fit(x_train, y_train)
y_predLogistic = logistic_reg.predict(x_test)

cm_logistic = confusion_matrix(y_test, y_predLogistic)
print("-----Logistic Regression Results-----")
print(cm_logistic)

logistic_accuracy = accuracy_score(y_test, y_predLogistic)
print("Accuracy of Logistic Regression")
print(logistic_accuracy)
print("")

#----------------KNN ALGORITHM----------------------
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski') 
knn.fit(x_train, y_train)
y_predKnn = knn.predict(x_test)

cm_knn = confusion_matrix(y_test, y_predKnn)
print("-----KNN Classifier Results-----")
print(cm_knn)

knn_accuracy = accuracy_score(y_test, y_predKnn)
print("Accuracy of KNN Classifier")
print(knn_accuracy)
print("")

#-------------SUPPORT VECTOR REGRESSION-------------
from sklearn.svm import SVC
svc = SVC(kernel = 'rbf')
svc.fit(x_train, y_train)
y_predSVC = svc.predict(x_test)

cm_svc = confusion_matrix(y_test, y_predSVC)
print("-----Support Vector Regression Results-----")
print(cm_svc)

svc_accuracy = accuracy_score(y_test, y_predSVC)
print("Accuracy of SVC Classifier")
print(svc_accuracy)
print("")

#-------------NAIF BAYES ALGORITHM------------------
from sklearn.naive_bayes import GaussianNB
naif_bayes = GaussianNB()
naif_bayes.fit(x_train, y_train)
y_predNB = naif_bayes.predict(x_test)

cm_bayes = confusion_matrix(y_test, y_predNB)
print("-----Naif Bayes Results-----")
print(cm_bayes)

bayes_accuracy = accuracy_score(y_test, y_predNB)
print("Accuracy of Naif Bayes")
print(bayes_accuracy)
print("")

#-------------DECISION TREE CLASSIFICATION-----------
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(criterion = 'entropy')
decision_tree.fit(x_train, y_train)
y_predTree = decision_tree.predict(x_test)

cm_tree = confusion_matrix(y_test, y_predTree)
print("-----Decision Tree Results-----")
print(cm_tree)

tree_accuracy = accuracy_score(y_test, y_predTree)
print("Accuracy of Decision Tree")
print(tree_accuracy)
print("")

#------------RANDOM FOREST CLASSIFICATION------------
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
random_forest.fit(x_train, y_train)
y_predForest = random_forest.predict(x_test)

cm_random = confusion_matrix(y_test, y_predForest)
print("-----Random Forest Results-----")
print(cm_random)

forest_accuracy = accuracy_score(y_test, y_predForest)
print("Accuracy of Random Forest Classifier")
print(forest_accuracy)
print("")