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
import read_files as rf
import tools as t

# Generate a ANN regressor
def build_net(n_input=None):

    #Fitting regressor to the Training set, create your regressor here
    regressor = Sequential()

    #Select the number of units
    n_in1 = 12
    n_in2 = 6
    n_in3 = 3
    n_out = 1

    activation1 = 'relu'
    activation2 = 'linear'
    activation3 = 'softmax'
    activation4 = 'selu'
    activation5 = 'tanh'

    regressor.add(Dense(units = n_input))
    regressor.add(Dense(units = n_in1, kernel_initializer='uniform', activation = activation1))
    regressor.add(Dense(units = n_in2, kernel_initializer='uniform', activation = activation2))
    regressor.add(Dense(units = n_in3, kernel_initializer='uniform', activation = activation2))
    #regressor.add(Dense(units = n_in1, kernel_initializer = 'he_uniform', activation = activation2))
    #regressor.add(Dense(units = n_in2, kernel_initializer = 'uniform', activation = activation1))
    #regressor.add(Dense(units = n_in2, kernel_initializer = 'uniform', activation = activation2))
    regressor.add(Dense(n_out))

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
    #opt = tf.keras.optimizers.SGD(lr=0.1)
    opt = tf.keras.optimizers.Adam(lr=0.02)

    #regressor.compile(optimizer=opt, loss='mse', metrics=['mae'])
    regressor.compile(optimizer=opt, loss='msle', metrics=['mae'])
    #regressor.compile(optimizer=opt, loss='mae', metrics=['mae'])

    #regressor.compile(optimizer='adam', loss='mse', metrics=['mae'])
    #regressor.compile(optimizer='rmsprop', loss='mse', metrics=['mae'], callbacks=[early_stop])
    #regressor.compile(optimizer='rmsprop', loss='msle', metrics=['mae', 'mse'])
    #print(regressor.optimizer.learning_rate)

    return regressor, activation1



'''
                            MAIN PROGRAM
'''


trainANN = True
#trainANN = False
ann_model_base_name = 'output/ann_model_'

#data = rf.read_lg_lgf(); file_type = 'lgf_'
#data = rf.read_lg_fullbox(); file_type = 'ahf_'
data = rf.read_lg_rs_fullbox(files=[0,35]); file_type = 'rs_'

'''
all_columns = ['M_M31', 'M_MW', 'R', 'Vrad', 'Vtan', 'Nsub_M31', 'Nsub_MW', 'Npart_M31', 'Npart_MW', 'Vmax_MW', 'Vmax_M31', 'lambda_MW',
       'lambda_M31', 'cNFW_MW', 'c_NFW_M31', 'Xc_LG', 'Yc_LG', 'Zc_LG', 'AngMom', 'Energy', 'x_32', 'y_32', 'z_32', 'l1_32', 'l2_32', 'l3_32', 'dens_32', 
       'x_64', 'y_64', 'z_64', 'l1_64', 'l2_64', 'l3_64', 'dens_64', 'x_128', 'y_128', 'z_128', 'l1_128', 'l2_128', 'l3_128', 'dens_128', 'Mtot', 'Mratio', 'l_tot_128']
'''

#grid = 32
#l1 = 'l1_' + str(grid); l2 = 'l2_' + str(grid); l3 = 'l3_' + str(grid)
#dens = 'dens_' + str(grid)
#l_tot = 'l_tot_' + str(grid)
#data[l_tot] = data[l1] + data[l2] + data[l3]

# Add some useful combinations to the dataframe
data['Mtot'] = data['M_M31'] + data['M_MW']
data['Mratio'] = data['M_M31'] / data['M_MW']
data['Vtot'] = np.sqrt(data['Vrad'] **2 + data['Vtan'] **2)

# Put some boundaries and refine selectin
vrad_max = -1.0
vtan_max = 1000.0
ratio_max = 4.0
r_max = 1300.0
r_min = 400.0
mass_min = 5.0e+11

data = t.filter_data(data=data, vrad_max=vrad_max, vtan_max=vtan_max, r_max=r_max, r_min=r_min, ratio_max=ratio_max)

# Manual normalization of the most important data types
#data = t.normalize_data(data=data)
data = t.equal_number_per_bin(data=data, n_bins=5, bin_col='M_MW')
#data = t.normalize_data(data=data, mode='log')
#data = t.normalize_data(data=data, mode='boxcox')
#data = t.normalize_data(data=data, mode='lin')
data = t.normalize_data(data=data, mode='mix2')

# SELECT COLUMNS FOR THE TRAINIG
#train_cols = ['Vrad', 'R']; pred_col = 'Mtot'; train_type = 'mass_total'
train_cols = ['Vrad', 'R', 'Vtan']; pred_col = 'Mtot'; train_type = 'mass_total'
#train_cols = ['Vrad', 'R', 'Vtan', 'Energy']; pred_col = 'Mtot'; train_type = 'mass_total'

#train_cols = ['Vrad', 'R', 'Vtan']; pred_col = 'M_MW_log'; train_type = 'mass_mw'
#train_cols = ['Vrad', 'R', 'Vtan']; pred_col = 'M_M31_log'; train_type = 'mass_m31'
#train_cols = ['Vrad', 'R', 'Vtan']; pred_col = 'Mratio'; train_type = 'mass_ratio'

#train_cols = ['Vrad', 'R', 'Mtot']; pred_col = 'Vtan'; train_type = 'vel_tan'
#train_cols = ['Vrad', 'R']; pred_col = 'Vtan'; train_type = 'vel_tan'

# Name of the ANN model to be saved
ann_name = ann_model_base_name + train_type + '.keras'

# Get the features and the desired output
X = data[train_cols].values
y = data[pred_col].values

if train_type == 'vel_tan':
    cols = ['Vtan_true', 'Vtan_pred']

elif train_type == 'mass_total':
    cols = ['Mtot_true', 'Mtot_pred']

elif train_type == 'mass_ratio':
    cols = ['Mratio_true', 'Mratio_pred']

print('Total train dataset sample size: ', len(X))

# Splitting the dataset into the Training set and Test set
test_size = 0.2
rand_state = 1234
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = rand_state)

#print('y_test: ')
#print(y_test)

# Feature Scaling - select scaling type
'''
#sc = StandardScaler()
#sc = MaxAbsScaler()
#sc = Normalizer()
sc = MinMaxScaler()

# Fit the scaler
sc.fit(X_train)
X_test = sc.transform(X_test)
X_train = sc.transform(X_train)
'''

#y_test = sc.transform(y_test)
#y_train = sc.transform(y_train)

n_input = len(train_cols)
regressor, act_func = build_net(n_input=n_input)

if trainANN == True:
    n_epochs = 5
    batch_size = 50

    # Fit the regressor to the training data and save it
    regressor.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size)
    regressor.save(ann_name)
    print('Regressor saved to: ', ann_name)

    predictions = regressor.predict(X_test)

    # FIXME TODO
    # TODO FIX THIS with new Keras / TensorFlow version
    # Load the ANN from a previously trained model
    #else:
    #print('Loading Regressor: ', ann_name)
    #regressor = ks.models.load_model(ann_name) 
    #predictions = regressor.predict(X_test)

    mae = metrics.mean_absolute_error(y_test, predictions)
    msq = metrics.mean_squared_error(y_test, predictions)
    mape = t.MAPE(y_test, predictions)

    print('MAE: ', mae, ' MSQ: ', np.sqrt(msq), ' MAPE: ', np.mean(mape) )

    data = pd.DataFrame() 
    data[cols[0]] = y_test
    data[cols[1]] = predictions
    data['ratio'] = np.log10(data[cols[1]] / data[cols[0]])

    sns.kdeplot(data[cols[0]], data[cols[1]], n_levels = 5)
    slope = np.polyfit(y_test, predictions, 1)
    print('Slope: ', slope)

    add_name = '_' + act_func + '_' + file_type

    for col in train_cols:
        add_name = add_name + '_' + col

    file_name = 'output/ann_dens_' + train_type + add_name + '.png'
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show(block=False)
    plt.pause(4)
    plt.close()
    plt.cla()
    plt.clf()
    
    plt.xlim(-0.5, 0.75)
    pct = t.percentiles(data['ratio'], perc=[20, 50, 80])

    file_name = 'output/ann_dist_' + train_type + add_name + '.png'
    sns.distplot(data['ratio'])
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show(block=False)
    plt.pause(4)
    plt.close()





