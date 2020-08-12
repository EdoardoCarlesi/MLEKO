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

# Put some threshold on LG properties from different datasets
ratio_max = 4.0
mass_min = 6.0e+11
vrad_max = -10
vtan_max = 500.0
r_max = 1200.0
r_min = 450.0

# Normalization constants
mass_norm = 1.0e+12
vel_norm = 100.0
r_norm = 1000.0

# Apply some restriction on data properties
filterData = True

# Read the different datasets
data_rs = rf.read_lg_rs_fullbox(files = [0, 10])
data_rs['Mtot'] = data_rs['M_M31'] + data_rs['M_MW']
data_rs['Mratio'] = data_rs['M_M31'] / data_rs['M_MW']
data_rs['Mlog'] = np.log10(data_rs['Mtot']/mass_norm)
data_rs['Vtot'] = np.sqrt(data_rs['Vrad'] **2 + data_rs['Vtan'] **2)
data_rs['Vlog'] = np.log10(data_rs['Vtot'])
data_rs['R_norm'] = data_rs['R'] / r_norm

data_ahf = rf.read_lg_fullbox()
data_ahf['Mtot'] = data_ahf['M_M31'] + data_ahf['M_MW']
data_ahf['Mratio'] = data_ahf['M_M31'] / data_ahf['M_MW']
data_ahf['Mlog'] = np.log10(data_ahf['Mtot']/mass_norm)
data_ahf['Vtot'] = np.sqrt(data_ahf['Vrad'] **2 + data_ahf['Vtan'] **2)
data_ahf['Vlog'] = np.log10(data_ahf['Vtot'])
data_ahf['R_norm'] = data_ahf['R'] / r_norm

data_lgf = rf.read_lg_lgf()
data_lgf['Mtot'] = data_lgf['M_M31'] + data_lgf['M_MW']
data_lgf['Mratio'] = data_lgf['M_M31'] / data_lgf['M_MW']
data_lgf['Mlog'] = np.log10(data_lgf['Mtot']/mass_norm)
data_lgf['Vtot'] = np.sqrt(data_lgf['Vrad'] **2 + data_lgf['Vtan'] **2)
data_lgf['Vlog'] = np.log10(data_lgf['Vtot'])
data_lgf['R_norm'] = data_lgf['R'] / r_norm

# Rescale the data
if filterData == True:
    data_lgf = data_lgf[data_lgf['Vrad'] < vrad_max]
    data_lgf = data_lgf[data_lgf['Vtan'] < vtan_max]
    data_lgf = data_lgf[data_lgf['R'] < r_max]
    data_lgf = data_lgf[data_lgf['R'] > r_min]
    data_lgf = data_lgf[data_lgf['M_MW'] > mass_min]
    data_lgf['Vrad_norm'] = np.log10(-data_lgf['Vrad'] / vel_norm)
    data_lgf['Vtan_norm'] = np.log10(data_lgf['Vtan'] / vel_norm)

    data_rs = data_rs[data_rs['Vrad'] < vrad_max]
    data_rs = data_rs[data_rs['Vtan'] < vtan_max]
    data_rs = data_rs[data_rs['R'] < r_max]
    data_rs = data_rs[data_rs['R'] > r_min]
    data_rs = data_rs[data_rs['M_MW'] > mass_min]
    data_rs['Vrad_norm'] = np.log10(-data_rs['Vrad'] / vel_norm)
    data_rs['Vtan_norm'] = np.log10(data_rs['Vtan'] / vel_norm)

    data_ahf = data_ahf[data_ahf['Vrad'] < vrad_max]
    data_ahf = data_ahf[data_ahf['Vtan'] < vtan_max]
    data_ahf = data_ahf[data_ahf['R'] < r_max]
    data_ahf = data_ahf[data_ahf['R'] > r_min]
    data_ahf = data_ahf[data_ahf['M_MW'] > mass_min]
    data_ahf['Vrad_norm'] = np.log10(-data_ahf['Vrad'] / vel_norm)
    data_ahf['Vtan_norm'] = np.log10(data_ahf['Vtan'] / vel_norm)

print('Data sample numbers. LGF: ', len(data_lgf), ' BigMD: ', len(data_rs), ' SmallBox: ', len(data_ahf))


# 1D Distributions
#col = 'R'
col = 'Mlog'
#col = 'Mratio'
#col = 'Vtan'; plt.xlim(0, 500.0)
#col = 'Vrad'; plt.xlim(-500.0, 190.0)
#col = 'Vrad_norm'
#col = 'Vtan_norm'

sns.distplot(data_lgf[col], color='blue', label='LGF')
sns.distplot(data_ahf[col], color='red', label='SmallBox')
sns.distplot(data_rs[col], color='green', label='BigMD')
plt.legend()

file_out = 'output/distplot_compare_' + col + '.png'
plt.savefig(file_out)

plt.clf()
plt.cla()

# 2D Distributions
col_x = 'Mlog'
#col_y = 'R_norm'
col_y = 'Vrad_norm'
#col_y = 'Vtan_norm'
n_levels = 4

# Linear fit
slope_lgf = np.polyfit(data_lgf[col_x], data_lgf[col_y], 1)
slope_rs = np.polyfit(data_rs[col_x], data_rs[col_y], 1)
slope_ahf = np.polyfit(data_ahf[col_x], data_ahf[col_y], 1)

x = [data_lgf[col_x].min(), data_lgf[col_x].max()]
y_lgf = [slope_lgf[0] * x[0] + slope_lgf[1], slope_lgf[0] * x[1] + slope_lgf[1]]
y_rs = [slope_rs[0] * x[0] + slope_rs[1], slope_rs[0] * x[1] + slope_rs[1]]
y_ahf = [slope_ahf[0] * x[0] + slope_ahf[1], slope_ahf[0] * x[1] + slope_ahf[1]]

sns.kdeplot(data_lgf[col_x], data_lgf[col_y], color='blue', label='LGF', n_levels = n_levels)
sns.kdeplot(data_ahf[col_x], data_ahf[col_y], color='red', label='SmallBox', n_levels = n_levels)
sns.kdeplot(data_rs[col_x], data_rs[col_y], color='green', label='BigMD', n_levels = n_levels)

sns.lineplot(x, y_lgf, color='blue')
sns.lineplot(x, y_rs, color='green')
sns.lineplot(x, y_ahf, color='red')

file_out = 'output/distplot_contours_' + col_x + '_' + col_y + '.png'
s_lgf = '%.3f' % slope_lgf[0]
s_rs = '%.3f' % slope_rs[0]
s_ahf = '%.3f' % slope_ahf[0]

i_lgf = '%.3f' % slope_lgf[1]
i_rs = '%.3f' % slope_rs[1]
i_ahf = '%.3f' % slope_ahf[1]

title  = ' --> SlopeLGF    : ' + s_lgf + ' SmallBox: ' + s_ahf + ' BigMD: ' + s_rs
title2 = ' --> InterceptLGF: ' + i_lgf + ' SmallBox: ' + i_ahf + ' BigMD: ' + i_rs
print(col_x, col_y, title)
print(col_x, col_y, title2)
plt.title(title)
plt.savefig(file_out)

#plt.show()






