'''
    CluesML
    Machine Learning tools for the CLUES project

    (C) Edoardo Carlesi 2020
    https://github.com/EdoardoCarlesi/CluesML
'''


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


sns.set_style('whitegrid')
data_train = 'dati/kyphosis.csv'

train = pd.read_csv(data_full)

print(train.info())
print(train.head())

#sns.pairplot(train, hue = 'Kyphosis')

#scaler = StandardScaler()
#logmod = LogisticRegression()
#knn = KNeighborsClassifier(n_neighbors = 37)
dtree = DecisionTreeClassifier()

X = train.drop('Kyphosis', axis=1)
y = train['Kyphosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 40)
dtree.fit(X_train, y_train)
pred = dtree.predict(X_test)

print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

print('RANDOM FOREST')

rforest = RandomForestClassifier(n_estimators = 200)
rforest.fit(X_train, y_train)
pred2 = rforest.predict(X_test)

print(classification_report(y_test, pred2))
print(confusion_matrix(y_test, pred2))


scaler.fit(train.drop('TARGET CLASS', axis = 1))
scaled_feat = scaler.transform(train.drop('TARGET CLASS', axis = 1))
data_scaled = pd.DataFrame(scaled_feat, columns = train.columns[:-1])

print(scaled_feat)

print(data_scaled.head())

pred = knn.predict(X_test)
pred2 = logmod.predict(X_test)
print(classification_report(y_test, pred))




