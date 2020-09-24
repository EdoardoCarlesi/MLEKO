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


def ta_mass(r, v):
    t0 = 14.75
    tam = ta.mass_estimate(r=r, v=v, t=t0)
    return tam


sns.set_style('whitegrid')

file_new_fb = '/home/edoardo/CLUES/PyRCODIO/output/lg_fb_new'
#data = rf.read_lg_fullbox_vweb(grids = [32, 64, 128])
#data = rf.read_lg_fullbox(TA=True); name_add = '_ahf'
data = rf.read_lg_fullbox(file_base=file_new_fb, TA=False, radii=True); name_add = '_new'
#data = rf.read_lg_rs_fullbox(files=[0, 10]); name_add = '_rs'
#data = rf.read_lg_lgf(TA=True); name_add = '_lgf'

all_columns = ['M_M31', 'M_MW', 'R', 'Vrad', 'Vtan', 'Nsub_M31', 'Nsub_MW', 'Npart_M31', 'Npart_MW', 'Vmax_MW', 'Vmax_M31', 'lambda_MW',
       'lambda_M31', 'cNFW_MW', 'c_NFW_M31', 'Xc_LG', 'Yc_LG', 'Zc_LG', 'AngMom', 'Energy', 'x_32', 'y_32', 'z_32', 'l1_32', 'l2_32', 'l3_32', 'dens_32', 
       'x_64', 'y_64', 'z_64', 'l1_64', 'l2_64', 'l3_64', 'dens_64', 'x_128', 'y_128', 'z_128', 'l1_128', 'l2_128', 'l3_128', 'dens_128', 'Mtot', 'Mratio', 'l_tot_128']

print(data.info())

grid = 128
mass_norm = 1.0e+12

l1 = 'l1_' + str(grid); l2 = 'l2_' + str(grid); l3 = 'l3_' + str(grid)
dens = 'dens_' + str(grid)
l_tot = 'l_tot_' + str(grid)

# Add some useful combinations to the dataframe
data = data[data['R_5000.0'] > 0.0]

data['Mtot'] = data['M_M31'] + data['M_MW']
data['Mratio'] = data['M_M31'] / data['M_MW']
data['Mlog'] = np.log10(data['Mtot']/mass_norm)
data['R_5000.0'] = np.log10(data['R_5000.0']/mass_norm)

mratio_max = 5.0
vrad_max = -1.0
vtan_max = 1000.0
r_max = 1200.0
r_min = 400.0
mass_min = 5.0e+11

data = data[data['Vrad'] < vrad_max]
print('Ndata after Vrad cut: ', len(data))
data = data[data['R'] < r_max]
print('Ndata after Rmax cut: ', len(data))
data = data[data['R'] > r_min]
print('Ndata after Rmin cut: ', len(data))
data = data[data['Vtan'] < vtan_max]
print('Ndata after Vtan cut: ', len(data))
data = data[data['Mratio'] < mratio_max]
print('Ndata after Mratio cut: ', len(data))
data = data[data['M_MW'] > mass_min]
print('Ndata after M_MW cut: ', len(data))

all_r = data['R'].values
all_v = data['Vrad'].values
all_tam = np.zeros((len(all_r)))

data['Vrad'] = np.log10(-data['Vrad'] / 100.0)

try:
    data['Mlog_TA'] = np.log10(data['M_TA'] / mass_norm)
except:
    print('No timing argument')

#data['Vtan'] = np.log(data['Vtan'] / 100.0)
#data['denslog'] = np.log10(data['dens_128'])
#data[l_tot] = data[l1] + data[l2] + data[l3]
#sns.scatterplot(x='Mlog', y='Vrad', data = data)
#plt.show()
#sns.lmplot(x=dens, y=l_tot, data=data)

# Set some parameters for the random forest and gradient boosted trees
n_estimators = 150
max_depth = 4
max_samples = 30
min_samples_split = 10
boot = False
n_jobs = 2
test_size = 0.2

regressor_type = 'random_forest'
#regressor_type = 'gradient_boost'
#regressor_type = 'linear'

# Regression type, feature selection and target variable
#train_cols = ['R','Vrad']; test_col = 'Mratio'; train_type = 'mass_ratio'
#train_cols = ['R','Vrad', 'M_MW']; test_col = 'Mratio'; train_type = 'mass_ratio'

#train_cols = ['Vrad', 'R']; test_col = 'Mlog'; train_type = 'mass_total'
#train_cols = ['R','Vrad', 'Vtan']; test_col = 'Mlog'; train_type = 'mass_total'
#train_cols = ['R','Vrad', 'R_6000.0']; test_col = 'Mlog'; train_type = 'mass_total'
#train_cols = ['R','Vrad', 'Vtan']; test_col = 'Mlog'; train_type = 'mass_total'
#train_cols = ['Vrad', 'R', 'Vtan', 'AngMom']; test_col = 'Mlog'; train_type = 'mass_total'
#train_cols = ['Vrad', 'R', 'Vtan', 'AngMom']; test_col = 'Mtot'; train_type = 'mass_total'
#train_cols = ['M_M31', 'M_MW', 'R', 'Vtan']; test_col = 'Mtot'; train_type = 'mass_total'
#train_cols = ['Vrad']; test_col = 'Mtot'; train_type = 'mass_total'
#train_cols = ['R','Vrad', 'Vtan', 'M_MW']; test_col = 'Mlog'; train_type = 'mass_total'
#train_cols = ['R','Vrad', 'Energy']; test_col = 'Mlog'; train_type = 'mass_total'
#train_cols = ['R','Vrad', 'Vtan']; test_col = 'Mtot'; train_type = 'mass_total'
#train_cols = ['R','Vrad', 'Vtan', l1, l2, l3, dens]; test_col = 'Mtot'; train_type = 'mass_total'
#train_cols = ['R','Vrad', 'AngMom', 'Vtan']; test_col = 'Mtot'; train_type = 'mass_total'
#train_cols = ['R','Vrad', 'Vtan', dens]; test_col = 'Mtot'; train_type = 'mass_total'
#train_cols = ['R','Vrad', dens]; test_col = 'Mtot'; train_type = 'mass_total'
#train_cols = ['R','Vrad', 'Vtan', dens]; test_col = 'Mtot'; train_type = 'mass_total'
#train_cols = ['R','Vrad', 'Vtan']; test_col = 'Mlog'; train_type = 'mass_total'
#train_cols = ['R','Vrad', 'Vtan']; test_col = 'Mlog'; train_type = 'mass_total'
#train_cols = ['M_M31','M_MW', 'Vtan']; test_col = 'Mtot'; train_type = 'mass_total'

train_cols = ['R', 'Vrad', 'R_8000.0']; test_col = 'Vtan'; train_type = 'vel_rad'
#train_cols = ['R','Vrad', 'Vtan']; test_col = 'M_M31'; train_type = 'mass_m31'
#train_cols = ['R','Vrad', 'Vtan']; test_col = 'M_M31'; train_type = 'mass_mw'

base_slope = np.polyfit(data[train_cols[0]], data[test_col], 1)
print('BaseSlope: ', base_slope)

new_col = 'M_LinFit'
data[new_col] = data[train_cols[0]].apply(lambda x: base_slope[0] * x + base_slope[1])

#print(data[new_col])
#print(data[test_col])
#sns.scatterplot(data[new_col], data[test_col])
#plt.show()

base_result_slope = np.polyfit(data[test_col], data[new_col], 1)
print('ResultSlope: ', base_result_slope)

# Select the regressor type
if regressor_type == 'random_forest':
    regressor = RandomForestRegressor(n_estimators=n_estimators, 
                                        max_depth=max_depth, 
                                        min_samples_split=min_samples_split, 
                                        #criterion=criterion,
                                        bootstrap=boot,
                                        max_samples=max_samples,
                                        n_jobs=n_jobs)

elif regressor_type == 'gradient_boost':
    regressor = GradientBoostingRegressor(n_estimators = n_estimators, 
                                            max_depth = max_depth,
                                            min_samples_split = min_samples_split)
elif regressor_type == 'linear':
    regressor = LinearRegression()

reg_name = regressor_type + '_' + train_type

#data['AngMom'] = data['AngMom'].apply(lambda x: np.log10(x))
#data['AngMom'].hist(bins=100)
#plt.show()

'''
# Do a PCA to check the data
pca_percent = 0.9
#pca_percent = None
#pca_cols = all_columns
pca_cols = ['R','Vrad', 'Vtan'] #, 'AngMom', 'Energy'] #, l1, l2, l3, dens]
data_pca = t.data_pca(data=data, columns=pca_cols, pca_percent=pca_percent)
print('PCA at ', pca_percent, ' n_components: ', len(data_pca.columns), ' n_original: ', len(all_columns))
print(data_pca.info())
print(data_pca.head())
'''

# Select the features for the training and test set
X = data[train_cols]
y = data[test_col]

try:
    z = data['Mlog_TA']
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = test_size, random_state = 42)
except:
    'No timing argument'

print('Total size: ', X.count())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)

print('Train size: ', len(X_train))
print('Test size: ', len(X_test))

scaler = MinMaxScaler()
#scaler = StandardScaler()

# This only optimizes the parameters to perform the scaling later on
scaler.fit(X_train)

#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

mae = metrics.mean_absolute_error(y_test, predictions)
msq = metrics.mean_squared_error(y_test, predictions)
mape = t.MAPE(y_test, predictions)

slope = np.polyfit(y_test, predictions, 1)

try:
    slope_ta = np.polyfit(z_test, predictions, 1)
    slope_ta_pred = np.polyfit(z_test, y_test, 1)
    print('Slope TA: ', slope_ta)
    print('Slope TA pred: ', slope_ta_pred)
except:
    'No timing argument'


if train_type == 'mass_total':
    cols = ['M_tot_true', 'M_tot_pred', 'M_tot_TA']

elif train_type == 'mass_ratio':
    cols = ['M_ratio_true', 'M_ratio_pred']
 
elif train_type == 'vel_rad':
    cols = ['V_rad_true', 'V_rad_pred']

data = pd.DataFrame() 
data[cols[0]] = y_test
data[cols[1]] = predictions

yy = []
x = [data[cols[0]].min(), data[cols[1]].max()]

for xx in x:
    yy.append(slope[0] * xx + slope[1])

print('MAE: ', mae, ' MSQ: ', np.sqrt(msq), ' MAPE: ', np.mean(mape) )
    
feat_title = ''

for feat in train_cols:
    feat_title = feat_title + '_' + feat

reg_name = reg_name + feat_title
title = name_add + feat_title + ' slope= ' + '%5.3f' % slope[0]

plt.figure(figsize=(5, 5))
sns.kdeplot(data[cols[0]], data[cols[1]])
sns.scatterplot(data[cols[0]], data[cols[1]]) #, n_levels = 4)
sns.lineplot(x, x)
sns.lineplot(x, yy)
plt.title(title)

print('Slope: ', slope)

try:
    importances = regressor.feature_importances_
    print('Feature importance:')
    print(X.columns)
    print(importances)

except:
    print('No feature importance')
    
#plt.scatterplot(y_test, predictions)
plt.tight_layout()

file_output ='output/' + reg_name + name_add + '.png' 
print('savefig to: ', file_output)
plt.savefig(file_output)
