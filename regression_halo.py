'''
    CluesML
    Machine Learning tools for the CLUES project

    (C) Edoardo Carlesi 2020
    https://github.com/EdoardoCarlesi/CluesML
'''

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import read_files as rf
import tools as t


sns.set_style('whitegrid')

data = rf.read_lg_fullbox_vweb(grids = [32, 64, 128])

all_columns = ['M_M31', 'M_MW', 'R', 'Vrad', 'Vtan', 'Nsub_M31', 'Nsub_MW', 'Npart_M31', 'Npart_MW', 'Vmax_MW', 'Vmax_M31', 'lambda_MW',
       'lambda_M31', 'cNFW_MW', 'c_NFW_M31', 'Xc_LG', 'Yc_LG', 'Zc_LG', 'AngMom', 'Energy', 'x_32', 'y_32', 'z_32', 'l1_32', 'l2_32', 'l3_32', 'dens_32', 
       'x_64', 'y_64', 'z_64', 'l1_64', 'l2_64', 'l3_64', 'dens_64', 'x_128', 'y_128', 'z_128', 'l1_128', 'l2_128', 'l3_128', 'dens_128', 'Mtot', 'Mratio', 'l_tot_128']

print(data.info())

grid = 128

l1 = 'l1_' + str(grid); l2 = 'l2_' + str(grid); l3 = 'l3_' + str(grid)
dens = 'dens_' + str(grid)
l_tot = 'l_tot_' + str(grid)

# Add some useful combinations to the dataframe
data['Mtot'] = data['M_M31'] + data['M_MW']
data['Mratio'] = data['M_M31'] / data['M_MW']
data[l_tot] = data[l1] + data[l2] + data[l3]

#sns.lmplot(x=dens, y=l_tot, data=data)
#plt.show()

# Set some parameters for the random forest and gradient boosted trees
n_estimators = 1000
max_depth = 4
min_samples_split = 12

# Regression type, feature selection and target variable
#train_cols = ['R','Vrad', 'Mtot']; test_col = 'Mratio'; train_type = 'mass_ratio_lambda'
#train_cols = ['R','Vrad', 'Mtot']; test_col = 'Mratio'; train_type = 'mass_ratio'
#train_cols = ['R','Vrad', 'Vtan', l1, l2, l3, dens]; test_col = 'Mtot'; train_type = 'mass_total_lambda' + str(grid)
#train_cols = ['R','Vrad', 'AngMom', 'Vtan']; test_col = 'Mtot'; train_type = 'mass_total'
train_cols = ['R','Vrad', 'Vtan']; test_col = 'Mtot'; train_type = 'mass_total'
#train_cols = ['R','Vrad', 'Vtan', dens]; test_col = 'Mtot'; train_type = 'mass_total'
#train_cols = ['R','Vrad', dens]; test_col = 'Mtot'; train_type = 'mass_total'
#train_cols = ['R','Vrad', 'Vtan', dens]; test_col = 'Mtot'; train_type = 'mass_total'
#train_cols = ['R','Vrad', 'Vtan']; test_col = 'Mtot'; train_type = 'mass_total'
#train_cols = ['Mtot','R', 'Vtan']; test_col = 'Vrad'; train_type = 'vel_rad'
#train_cols = ['R','Vrad', 'Vtan']; test_col = 'M_M31'; train_type = 'mass_m31'
#train_cols = ['R','Vrad', 'Vtan']; test_col = 'M_M31'; train_type = 'mass_mw'

# Select the regressor type
#regressor = LinearRegression(); reg_name = 'linear_reg' + '_' + train_type
regressor = RandomForestRegressor(n_estimators = n_estimators); reg_name = 'randomforest_reg' + '_' + train_type
#regressor = GradientBoostingRegressor(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split); reg_name = 'gradientboost_reg' + '_' + train_type

#data['AngMom'] = data['AngMom'].apply(lambda x: np.log10(x))
#data['AngMom'].hist(bins=100)
#plt.show()

# Do a PCA to check the data
pca_percent = 0.9
#pca_percent = None
pca_cols = all_columns
#pca_cols = ['R','Vrad', 'Vtan', 'AngMom', 'Energy', l1, l2, l3, dens]
data_pca = t.data_pca(data=data, columns=pca_cols, pca_percent=pca_percent)

print('PCA at ', pca_percent, ' n_components: ', len(data_pca.columns), ' n_original: ', len(all_columns))

print(data_pca.info())
print(data_pca.head())

# Select the features for the training and test set
X = data[train_cols]
y = data[test_col]

print('Total size: ', X.count())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 40)

print('Train size: ', len(X_train))
print('Test size: ', len(X_test))

#scaler = MinMaxScaler()
scaler = StandardScaler()

# This only optimizes the parameters to perform the scaling later on
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

#print(y_test.shape)

mae = metrics.mean_absolute_error(y_test, predictions)
msq = metrics.mean_squared_error(y_test, predictions)
mape = t.MAPE(y_test, predictions)

if train_type == 'mass_total' or train_type == 'mass_total_lambda' + str(grid): 
    print('MAE: ', mae/1.0e+12, ' MSQ: ', np.sqrt(msq)/1.0e+12, ' MAPE: ', np.mean(mape) )
    cols = ['M_tot_true', 'M_tot_pred']
    data = pd.DataFrame() 
    slope = np.polyfit(np.log10(y_test), np.log10(predictions), 1)
    
    # TODO check why one prediction is negative???
    #predictions = abs(predictions)
    data[cols[0]] = np.log10(y_test)
    data[cols[1]] = np.log10(predictions)
    x = [12.1, 12.5, 12.85]
    
    feat_title = '_'

    for feat in train_cols:
        feat_title = feat_title + '_' + feat

    reg_name = reg_name + feat_title
    title = feat_title + ' slope= ' + '%5.3f' % slope[0]

    sns.lmplot(x=cols[0], y=cols[1], data=data)
    sns.lineplot(x, x)
    plt.title(title)

    print('Slope: ', slope)

elif train_type == 'mass_m31':
    print('MAE: ', mae/1.0e+12, ' MSQ: ', np.sqrt(msq)/1.0e+12, ' MAPE: ', np.mean(mape) )
    ax = sns.scatterplot(x=np.log10(y_test), y=np.log10(predictions))
    cols = ['M_M31_true', 'M_M31_pred']
    data = pd.DataFrame() 
    data[cols[0]] = np.log10(y_test)
    data[cols[1]] = np.log10(predictions)
    sns.lmplot(x=cols[0], y=cols[1], data=data)

    slope = np.polyfit(np.log10(y_test), np.log10(predictions), 1)
    print('Slope: ', slope)

elif train_type == 'mass_mw':
    print('MAE: ', mae/1.0e+12, ' MSQ: ', np.sqrt(msq)/1.0e+12, ' MAPE: ', np.mean(mape) )
    cols = ['M_MW_true', 'M_MW_pred']
    data = pd.DataFrame() 
    data[cols[0]] = np.log10(y_test)
    data[cols[1]] = np.log10(predictions)
    sns.lmplot(x=cols[0], y=cols[1], data=data)

    slope = np.polyfit(np.log10(y_test), np.log10(predictions), 1)
    print('Slope: ', slope)

elif train_type == 'mass_ratio':
    print('MAE: ', mae, ' MSQ: ', np.sqrt(msq), ' MAPE: ', np.mean(mape) )
    cols = ['M_ratio_true', 'M_ratio_pred']
    ax = sns.scatterplot(y_test, predictions)
    ax.set(xlabel='M_ratio (true)', ylabel='M_ratio (pred)')

elif train_type == 'vel_rad':
    print('MAE: ', mae/100.0, ' MSQ: ', np.sqrt(msq)/100.0, ' MAPE: ', np.mean(mape) )
    ax = sns.scatterplot(y_test, predictions)

    slope = np.polyfit(y_test, predictions, 1)
    print('Slope: ', slope)
    ax.set(xlabel='V_rad (true)', ylabel='V_rad (pred)')

if reg_name == 'randomforest_reg' + '_' + train_type:
    importances = regressor.feature_importances_
    print(X.columns)
    print(importances)

#plt.scatterplot(y_test, predictions)
plt.tight_layout()
print('savefig to output/' + reg_name + '.png')
plt.savefig('output/' + reg_name + '.png')
