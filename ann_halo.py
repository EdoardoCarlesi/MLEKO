'''
    CluesML
    Machine Learning tools for the CLUES project

    (C) Edoardo Carlesi 2020
    https://github.com/EdoardoCarlesi/CluesML
'''

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import keras as ks
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tools as t

trainANN = True
#trainANN = False
ann_model_base_name = 'output/ann_model_'

sns.set_style('whitegrid')
data_00 = '/home/edoardo/CLUES/PyRCODIO/output/lg_fullbox_00.csv'
train_00 = pd.read_csv(data_00)
data_01 = '/home/edoardo/CLUES/PyRCODIO/output/lg_fullbox_01.csv'
train_01 = pd.read_csv(data_01)
data_02 = '/home/edoardo/CLUES/PyRCODIO/output/lg_fullbox_02.csv'
train_02 = pd.read_csv(data_02)
data_03 = '/home/edoardo/CLUES/PyRCODIO/output/lg_fullbox_03.csv'
train_03 = pd.read_csv(data_03)
data_04 = '/home/edoardo/CLUES/PyRCODIO/output/lg_fullbox_04.csv'
train_04 = pd.read_csv(data_04)

data = pd.concat([train_00, train_01, train_02, train_03, train_04])

data['Mtot'] = data['M_M31'] + data['M_MW']
data['Mratio'] = data['M_M31'] / data['M_MW']
data['VR'] = data['Vrad'] * data['R']
data['Vtot'] = np.sqrt(data['Vrad'] **2 + data['Vtan'] **2)

#train_cols = ['R','Vrad', 'Mtot']; pred_col = 'Mratio'; train_type = 'mass_ratio'
#train_cols = ['R','Vrad']; pred_col = 'Mtot'; train_type = 'mass_total'
#train_cols = ['R', 'Vrad', 'Vtan']; pred_col = 'Mtot'; train_type = 'mass_total'
#train_cols = ['VR', 'R', 'Vrad', 'Vtan', 'Vtot']; pred_col = 'Mtot'; train_type = 'mass_total'
#train_cols = ['R', 'Vrad', 'Vtan']; pred_col = 'Mtot'; train_type = 'mass_total'
#train_cols = ['Mratio', 'M_M31']; pred_col = 'Mtot'; train_type = 'mass_total'
#train_cols = ['M_MW', 'M_M31']; pred_col = 'Mtot'; train_type = 'mass_total'
#train_cols = ['R', 'Vrad', 'Vtan']; pred_col = 'Mtot'; train_type = 'mass_total'
train_cols = ['R', 'Vrad']; pred_col = 'Mtot'; train_type = 'mass_total'
#train_cols = ['R', 'M_MW', 'Vtan']; pred_col = 'Mtot'; train_type = 'mass_total'
#train_cols = ['M_M31', 'R','Vrad', 'Vtan']; pred_col = 'Mtot'; train_type = 'mass_total'
#train_cols = ['R','Vrad']; pred_col = 'Mtot'; train_type = 'mass_total'
#train_cols = ['R','Vrad', 'Vtan']; pred_col = 'M_M31'; train_type = 'mass_m31'
#train_cols = ['R','Mtot', 'Vtan']; pred_col = 'Vrad'; train_type = 'vrad'


'''
vtan_max = 150.0
data = data[data['Vtan'] < vtan_max]
'''
ann_name = ann_model_base_name + train_type + '.keras'

# Properly rescale all the units
#vrad_max = -150.0
vrad_max = data['Vrad'].min()
#data = data[data['Vrad'] > vrad_max]
#data['Vrad'] = (data['Vrad'] - vrad_max) / 100.0
data['Vrad'] = np.log(-(data['Vrad'] + vrad_max) / 100.0)

data['Mtot'] = np.log10(data['Mtot']/ 1.0e+12)
data['M_MW'] = np.log10(data['M_MW']/ 1.0e+12)
data['M_M31'] = np.log10(data['M_M31']/ 1.0e+12)

'''
# Normalize stuff manually and see what happens
for col in train_cols:
    data[col] = (data[col] - data[col].min())/(data[col].max() - data[col].min()) 
    #data[col] = data[col]/data[col].max()
    data[col].plot.hist(bins=50)
    #plt.show()
'''

#data['R'].plot.hist(bins=50)
#plt.show()

#data['Vrad'] = np.log((data['Vrad'] - vrad_max)) # / 100.0
#data['Vrad'] = np.power((data['Vrad'] - vrad_max), 0.3)
#data['Vtan'] = np.log(data['Vtan'])

#sns.pairplot(data[train_cols])
#plt.show()

print(data[train_cols].head())

X = data[train_cols].values
y = data[pred_col].values

if train_type == 'vrad':
    cols = ['V_rad_true', 'V_rad_pred']

elif train_type == 'mass_total':
    cols = ['M_tot_true', 'M_tot_pred']

print('Total train size: ', len(X))

# Splitting the dataset into the Training set and Test set
test_size = 0.2
rand_state = 1234
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = rand_state)

#print(y_train)

# Feature Scaling
sc = StandardScaler()
#sc = MaxAbsScaler()
#sc = Normalizer()
#sc = MinMaxScaler()
sc.fit(X_train)

X_test = sc.transform(X_test)
X_train = sc.transform(X_train)

'''
print(X_train.shape)
print(np.median(X_train[:, 0]))
print(np.median(X_train[:, 1]))
print(np.median(X_train[:, 2]))
'''

#Fitting regressor to the Training set, create your regressor here
regressor = Sequential()
#print(X_test)

#Select the number of units
n_in1 = 12
n_in2 = 4
n_out = 1
n_epochs = 12
batch_size = 30
n_input = len(train_cols)

activation1 = 'relu'
activation1 = 'softmax'
activation1 = 'selu'
activation1 = 'linear'

'''
activation1 = 'tanh'
'''

regressor.add(Dense(n_input, activation = activation1))
#regressor.add(Dense(n_in1, activation = activation1))
regressor.add(Dense(units = n_in1, kernel_initializer = 'uniform', activation = activation1))
#regressor.add(Dense(units = n_in1, kernel_initializer = 'he_uniform', activation = activation1))
#regressor.add(Dense(units = n_in1, kernel_initializer = 'he_uniform', activation = activation1))
#regressor.add(Dense(units = n_in1, kernel_initializer = 'he_uniform', activation = activation1))
#regressor.add(Dense(n_in2, activation = activation1))
regressor.add(Dense(units = n_out, kernel_initializer = 'uniform', activation = 'linear'))
#regressor.add(Dense(n_out))

'''
early_stop = EarlyStopping(monitor='loss', mode="min", verbose = 1, patience = 10) 

# Input layer
regressor.add(Dense(units = n_in, kernel_initializer = 'he_uniform', activation = activation , input_dim = n_input))

# First hidden layer
#regressor.add(Dropout(0.2))

# Second hidden layer
#regressor.add(Dense(units = n_in2, kernel_initializer = 'uniform', activation = 'relu'))
#regressor.add(Dropout(0.2))

# Output layer, use linear activation function for regression problems
'''

#opt='adam'
#opt = tf.keras.optimizers.SGD(lr=0.008)
#opt = tf.keras.optimizers.SGD(lr=0.008)
opt = tf.keras.optimizers.Adam(lr=0.015)

regressor.compile(optimizer=opt, loss='mse', metrics=['mae'])
#regressor.compile(optimizer=opt, loss='msle', metrics=['mae'])
#regressor.compile(optimizer=opt, loss='mae', metrics=['mae'])

#K.set_value(
#regressor.optimizer.learning_rate = 0.1
#K.set_value(regressor.optimizer.learning_rate, 0.1)
#regressor.compile(optimizer='adam', loss='mse', metrics=['mae'])
#regressor.compile(optimizer='rmsprop', loss='mse', metrics=['mae'], callbacks=[early_stop])
#regressor.compile(optimizer='rmsprop', loss='msle', metrics=['mae', 'mse'])

print(regressor.optimizer.learning_rate)

if trainANN == True:
    # Fit the regressor to the training data and save it
    regressor.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size)
    regressor.save(ann_name)
    print('Regressor saved to: ', ann_name)

    predictions = regressor.predict(X_test)

    '''
# TODO FIX THIS with new Keras / TensorFlow version
# Load the ANN from a previously trained model
else:

    print('Loading Regressor: ', ann_name)
    regressor = ks.models.load_model(ann_name) 
    predictions = regressor.predict(X_test)
    '''

    mae = metrics.mean_absolute_error(y_test, predictions)
    msq = metrics.mean_squared_error(y_test, predictions)
    mape = t.MAPE(y_test, predictions)

    print('MAE: ', mae, ' MSQ: ', np.sqrt(msq), ' MAPE: ', np.mean(mape) )

    data = pd.DataFrame() 
    data[cols[0]] = y_test
    data[cols[1]] = predictions
    #sns.lineplot(x=cols[0], y=cols[0], data=data)
    sns.lmplot(x=cols[0], y=cols[1], data=data)
    slope = np.polyfit(y_test, predictions, 1)
    print('Slope: ', slope)
    file_name = 'output/ann_' + train_type + '.png'
    plt.savefig(file_name)
    #plt.show()     
    



