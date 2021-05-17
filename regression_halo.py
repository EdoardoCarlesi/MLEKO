"""
    MLEKO
    Machine Learning Environment for KOsmology

    (C) Edoardo Carlesi 2020
    https://github.com/EdoardoCarlesi/MLEKO
"""


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
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
import pickle

import data_visualization as dv
import montecarlo as mc

sns.set_style('whitegrid')

do_mc = True
#data = rf.read_lg_fullbox_vweb(grids = [32, 64, 128])

data = rf.read_lg_fullbox(TA=False); name_add = '_sb'
#data = rf.read_lg_rs_fullbox(files=[0, 20]); name_add = '_rs'
#data = rf.read_lg_lgf(TA=True); name_add = '_lgf'

all_columns = ['M_M31', 'M_MW', 'R', 'Vrad', 'Vtan', 'Nsub_M31', 'Nsub_MW', 'Npart_M31', 'Npart_MW', 'Vmax_MW', 'Vmax_M31', 'lambda_MW',
       'lambda_M31', 'cNFW_MW', 'c_NFW_M31', 'Xc_LG', 'Yc_LG', 'Zc_LG', 'AngMom', 'Energy', 'x_32', 'y_32', 'z_32', 'l1_32', 'l2_32', 'l3_32', 'dens_32', 
       'x_64', 'y_64', 'z_64', 'l1_64', 'l2_64', 'l3_64', 'dens_64', 'x_128', 'y_128', 'z_128', 'l1_128', 'l2_128', 'l3_128', 'dens_128', 'Mtot', 'Mratio', 'l_tot_128']

print(data.info())

grid = 128

l1 = 'l1_' + str(grid); l2 = 'l2_' + str(grid); l3 = 'l3_' + str(grid)
dens = 'dens_' + str(grid)
l_tot = 'l_tot_' + str(grid)

# Best parameter set SmallBox
if name_add == '_sb':
    mratio_max = 5.0
    vrad_max = -10.0
    vtan_max = 1000.0
    r_max = 1400.0
    r_min = 400.0
    mass_min = 5.0e+11

# Best parameter set LGF
if name_add == '_lgf':
    mratio_max = 60.0
    vrad_max = -1.0
    vtan_max = 1000.0
    r_max = 1500.0
    r_min = 300.0
    mass_min = 5.0e+11

# Best parameter set RS
if name_add == '_rs':
    mratio_max = 5.0
    vrad_max = -10.0
    vtan_max = 500.0
    r_max = 1400.0
    r_min = 500.0
    mass_min = 5.0e+11

# Set some parameters for the random forest and gradient boosted trees
boot = False

# n_estimators = 200 # RandomForest
n_estimators = 250

# Make it high for random forest, small for gradient boosting
#max_depth = 50 # Decision Tree
#max_depth = 120 # Random Forest
max_depth = 10 # Gradient boost

max_samples = 500
max_features = 2
min_samples_split = 5
min_samples_leaf = 5
n_jobs = 2
n_bins = 20
test_size = 0.2

# Normalization factors
mass_norm = 1.0e+12
vel_norm = 100.0
r_norm = 1000.0

# Add some useful combinations to the dataframe
data['Mtot'] = data['M_M31'] + data['M_MW']
data['Mratio'] = data['M_M31'] / data['M_MW']
data['Mlog'] = np.log10(data['Mtot']/mass_norm)
data['M_MW_log'] = np.log10(data['M_MW']/mass_norm)
data['M_M31_log'] = np.log10(data['M_M31']/mass_norm)

# Refine selection
data = data[data['Vrad'] < vrad_max]
print('Ndata after Vrad cut: ', len(data))
data = data[data['R'] < r_max]
print('Ndata after Rmax cut: ', len(data))
data = data[data['R'] > r_min]
print('Ndata after Rmin cut: ', len(data))
data = data[data['Vtan'] < vtan_max]
print('Ndata after Vtan cut: ', len(data))
data = data[data['M_MW'] > mass_min]
print('Ndata after M_MW cut: ', len(data))
data = data[data['Mratio'] < mratio_max]
print('Ndata after Mratio cut: ', len(data))

try:
    data['Mlog_TA'] = np.log10(data['M_TA'] / mass_norm)
except:
    print('No timing argument mass')


# Do some data normalization
#data['R'] = data['R'] / r_norm
data['Vrad'] = np.log10(-data['Vrad'] / vel_norm)
data['Vtan'] = np.log(data['Vtan'] / vel_norm)
data['Vtot'] = np.sqrt(data['Vtan'].apply(lambda x: x*x) + data['Vrad'].apply(lambda x: x*x))
data['Ekin'] = 0.5 * data['Vtot'].apply(lambda x: x*x)

equal_label = ''
#data = dv.equal_number_per_bin(data=data, bin_col='Mlog', n_bins=10); equal_label = '_EQbin'

#data['denslog'] = np.log10(data['dens_128'])
#data[l_tot] = data[l1] + data[l2] + data[l3]
#sns.scatterplot(x='Mlog', y='Vrad', data = data)
#plt.show()
#sns.lmplot(x=dens, y=l_tot, data=data)

#regressor_type = 'linear'
#regressor_type = 'decision_tree'
#regressor_type = 'random_forest'
regressor_type = 'gradient_boost'

# Regression type, feature selection and target variable
#train_cols = ['Vrad', 'R']; pred_col = 'Mlog'; train_type = 'mass_total'
#train_cols = ['Vrad', 'R', 'Vtan', 'Energy']; pred_col = 'Mlog'; train_type = 'mass_total'

train_cols = ['Vrad', 'R', 'Vtan']; pred_col = 'Mlog'; train_type = 'mass_total'
#train_cols = ['Vrad', 'R', 'Vtan']; pred_col = 'M_MW_log'; train_type = 'mass_mw'
#train_cols = ['Vrad', 'R', 'Vtan']; pred_col = 'M_M31_log'; train_type = 'mass_m31'
#train_cols = ['Vrad', 'R', 'Vtan']; pred_col = 'Mratio'; train_type = 'mass_ratio'

#train_cols = ['Vrad', 'R', 'Mlog']; pred_col = 'Vtan'; train_type = 'vel_tan'
#train_cols = ['Vrad', 'R']; pred_col = 'Vtan'; train_type = 'vel_tan'

base_slope = np.polyfit(data[train_cols[0]], data[pred_col], 1)
print('BaseSlope: ', base_slope)

new_col = 'M_LinFit'
data[new_col] = data[train_cols[0]].apply(lambda x: base_slope[0] * x + base_slope[1])

#print(data[new_col])
#print(data[pred_col])
#sns.scatterplot(data[new_col], data[pred_col])
#plt.show()

base_result_slope = np.polyfit(data[pred_col], data[new_col], 1)
print('ResultSlope: ', base_result_slope)

# Select the regressor type
if regressor_type == 'random_forest':
    regressor = RandomForestRegressor(
                                        n_estimators=n_estimators, 
                                        max_depth=max_depth, 
                                        max_features=max_features,
                                        min_samples_split=min_samples_split, 
                                        min_samples_leaf=min_samples_leaf,
                                        bootstrap=boot,
                                        max_samples=max_samples,
                                        n_jobs=n_jobs)

elif regressor_type == 'gradient_boost':
    regressor = GradientBoostingRegressor(
                                            n_estimators=n_estimators, 
                                            max_depth=max_depth,
                                            max_features=max_features,
                                            min_samples_leaf=min_samples_leaf,
                                            min_samples_split=min_samples_split)
elif regressor_type == 'linear':
    regressor = LinearRegression()

elif regressor_type == 'decision_tree':
    regressor = DecisionTreeRegressor(
                                            max_features=max_features,
                                            min_samples_split=min_samples_split,
                                            max_depth=max_depth
                                    )

reg_name = regressor_type + '_' + train_type

# Select the features for the training and test set
X = data[train_cols]
y = data[pred_col]

try:
    z= data['Mlog_TA']
except:
    print('No timing argument mass')

print('Total size: ', X.count())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)

try:
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = test_size, random_state = 42)
except:
    'No TA'

print('Train size: ', len(X_train))
print('Test size: ', len(X_test))

'''
scaler = MinMaxScaler()
#scaler = StandardScaler()
# This only optimizes the parameters to perform the scaling later on
scaler.fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)
'''

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
    print('No timing argument mass')

col_ratio = 'pred_true_ratio'

if train_type == 'mass_total':
    cols = ['M_tot_true', 'M_tot_pred', 'M_tot_TA']

elif train_type == 'mass_ratio':
    cols = ['M_ratio_true', 'M_ratio_pred']
 
elif train_type == 'vel_rad':
    cols = ['V_rad_true', 'V_rad_pred']

elif train_type == 'vel_tan':
    cols = ['V_tan_true', 'V_tan_pred']

elif train_type == 'mass_m31':
    cols = ['M31_true', 'M31_pred']

elif train_type == 'mass_mw':
    cols = ['MW_true', 'MW_pred']


data = pd.DataFrame() 
data[cols[0]] = y_test
data[cols[1]] = predictions
data[col_ratio] = np.log10(predictions / y_test)

yy = []
x = [data[cols[0]].min(), data[cols[0]].max()]

for xx in x:
    yy.append(slope[0] * xx + slope[1])

print('MAE: ', mae, ' MSQ: ', np.sqrt(msq), ' MAPE: ', np.mean(mape) )
    
feat_title = ''

for feat in train_cols:
    feat_title = feat_title + '_' + feat

reg_name = reg_name + feat_title
title = 'Correlation' + feat_title + equal_label + ' slope= ' + '%5.3f' % slope[0]

# Plot the density levels
plt.grid(False)
plt.figure(figsize=(5, 5))
sns.kdeplot(data[cols[0]], data[cols[1]])
sns.scatterplot(data[cols[0]], data[cols[1]]) #, n_levels = 4)
sns.lineplot(x, x)
sns.lineplot(x, yy)
plt.title(title)
plt.tight_layout()
file_output ='output/' + reg_name + equal_label + name_add + '.png' 

print('savefig to: ', file_output)
plt.savefig(file_output)
plt.clf()
plt.cla()
print('Slope: ', slope)

# Make sure we don't stop the program if the regressor does not support feature importance
try:
    importances = regressor.feature_importances_
    print('Feature importance:')
    print(X.columns)
    print(importances)

except:
    print('No feature importance')

file_output_ratio ='output/ratio_' + reg_name + equal_label + name_add + '.png' 
sns.distplot(data[col_ratio], bins=n_bins)

data = data.dropna()
med = np.median(data[col_ratio])
std = np.std(data[col_ratio])
title = cols[1] + '_' + cols[0] + ' med: ' + '%5.3f' % med + ', std: %5.3f' % std

print('Ratio, median=', med, ' std=', std)
print('Saving ratio to:', file_output_ratio, ' median: ', med, ' stddev: ', std)

plt.title(title)
plt.savefig(file_output_ratio)
plt.clf()
plt.cla()

regressor_file = 'output/regressor_' + regressor_type + '_' + train_type + equal_label + name_add + '_model.pkl'
print('Saving model to: ', regressor_file)
pickle.dump(regressor, open(regressor_file, 'wb'))

if do_mc == True:
    print('Montecarlo methods...')

    n_pts=10000
    #cols=['Vrad', 'R', 'Vtan']

    cols=train_cols

    if train_type == 'vel_tan':
        vrad = np.log10([1.00, 1.20])
        mtot = np.log10([1.00, 5.00])
        df_mc = mc.gen_mc(
                    distribution='gauss', 
                    n_pts=n_pts, 
                    cols=cols,
                    vrad=vrad, 
                    rad=[450, 550],
                    mtot=mtot) 
    else:

        #vrad = np.log10([1.00, 1.20])
        #vtan = np.log10([0.01, 2.00])
        vrad = [105, 115]
        vtan = [22, 92]
        rad = [490, 550]
        df_mc = mc.gen_mc(
                    distribution='gauss', 
                    n_pts=n_pts, 
                    cols=cols,
                    vrad=vrad, 
                    rad=rad, 
                    vtan=vtan) 

    name_add = pred_col 
    mc.plot_mc_simple(mc_df=df_mc,
                    extra_info=name_add+equal_label, 
                    show=True, 
                    cols=cols,
                    n_bins=n_bins,
                    train_type=train_type,
                    title_add=pred_col,
                    regressor_type=regressor_type,
                    regressor_file=regressor_file)

    print('Done.')
