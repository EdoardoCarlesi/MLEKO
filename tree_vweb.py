'''
    CluesML
    Machine Learning tools for the CLUES project

    (C) Edoardo Carlesi 2020
    https://github.com/EdoardoCarlesi/CluesML
'''

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tools as t

sns.set_style('whitegrid')
data_train_00 = '/home/edoardo/CLUES/PyRCODIO/output/lg_fullbox_00.csv'
train_00 = pd.read_csv(data_train_00)
data_train_01 = '/home/edoardo/CLUES/PyRCODIO/output/lg_fullbox_01.csv'
train_01 = pd.read_csv(data_train_01)
data_train_02 = '/home/edoardo/CLUES/PyRCODIO/output/lg_fullbox_02.csv'
train_02 = pd.read_csv(data_train_02)
data_train_03 = '/home/edoardo/CLUES/PyRCODIO/output/lg_fullbox_03.csv'
train_03 = pd.read_csv(data_train_03)

train = pd.concat([train_00, train_01, train_02, train_03])

data_test = '/home/edoardo/CLUES/PyRCODIO/output/lg_fullbox_04.csv'
train['Mtot'] = train['M_M31'] + train['M_MW']
train['Mratio'] = train['M_M31'] / train['M_MW']

#print(train['Mratio'])

train.drop(['sub_code', 'simu_code', 'Nsub_M31', 'Nsub_MW', 'Xc_LG', 'Yc_LG', 'Zc_LG'], axis=1, inplace=True)
train.drop(['cNFW_MW', 'c_NFW_M31', 'Vmax_MW', 'Vmax_M31'], axis=1, inplace=True)
train.drop(['Npart_MW', 'Npart_M31'], axis=1, inplace=True)
train.drop(['lambda_MW', 'lambda_M31'], axis=1, inplace=True)
print(train.info())
print(train.head())
#sns.pairplot(train)

#regressor = LinearRegression(); reg_name = 'linreg'
#regressor = LogisticRegression(); reg_name = 'logreg'
regressor = RandomForestRegressor(); reg_name = 'randomforest_reg'

X = train.drop(['Mtot', 'M_MW', 'M_M31', 'Mratio'], axis=1)
y = train['Mtot']

print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 40)

#scaler = MinMaxScaler()
scaler = StandardScaler()

# This only optimizes the parameters to perform the scaling later on
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

mae = metrics.mean_absolute_error(y_test, predictions)
msq = metrics.mean_squared_error(y_test, predictions)
mape = t.MAPE(y_test, predictions)

print('MAE: ', mae/1.0e+12, ' MSQ: ', np.sqrt(msq)/1.0e+12, ' MAPE: ', np.mean(mape) )
sns.scatterplot(y_test, predictions)


'''
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
'''
plt.show()


