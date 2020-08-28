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


def rescale_data(data=None, mass_norm=1.0e+12, r_norm=1000.0):

    data['Mtot'] = data['M_M31'] + data['M_MW']
    data['Mratio'] = data['M_M31'] / data['M_MW']
    data['Mlog'] = np.log10(data['Mtot']/mass_norm)
    data['Vtot'] = np.sqrt(data['Vrad'] **2 + data['Vtan'] **2)
    data['Vlog'] = np.log10(data['Vtot'])
    data['R_norm'] = data['R'] / r_norm
    data['Vrad_norm'] = np.log10(-data['Vrad'] / vel_norm)
    data['Vtan_norm'] = np.log10(data['Vtan'] / vel_norm)

    return data

def filter_data(data=None, vrad_max=-1.0, vrad_min=-200.0, r_max=1500.0, r_min=350.0, mass_min=5.0e+11, vel_norm=100.0, vtan_max=500.0):

    data = data[data['Vrad'] < vrad_max]
    data = data[data['Vrad'] > vrad_min]
    data = data[data['R'] < r_max]
    data = data[data['R'] > r_min]
    data = data[data['Vtan'] < vtan_max]
    data = data[data['M_MW'] > mass_min]
    
    return data


def equal_number_per_bin(data=None, n_bins=10, bin_col='Mvir'):
    
    bins = np.zeros((n_bins))

    bins[0] = data[bin_col].min()
    bins[n_bins-1] = data[bin_col].max()
    bin_size = (bins[n_bins-1] - bins[0]) / float(n_bins)
 
    for i in range(1, n_bins-1):
        bins[i] = bin_size * i


    return new_data


def radial_velocity_binning(use_simu=None, data=None, vrad_max=vrad_max, vrad_min=vrad_min, 
                           r_max=r_max, r_min=r_min, vel_norm=vel_norm, mass_min=mass_min, vtan_max=vtan_max):

    print('Data sample ', use_simu, ' total datapoints: ', len(data))

    v_step = 20.0
    vrad_bins = []
    vrad_labels = []
    vrad_bins.append(0.0)

    for i in range(0, 9):
        vrad_max = -i * v_step
        vrad_min = -(i +1) * v_step

        data_new = filter_data(data=data, vrad_max=vrad_max, vrad_min=vrad_min, r_max=r_max, r_min=r_min, vel_norm=vel_norm, mass_min=mass_min, vtan_max=vtan_max)

        percentiles = np.percentile(data_new['Mtot'], [20, 50, 80])

        print('(', vrad_max, ', ', vrad_min, ') sample: ', len(data_new), 'percentiles: ', percentiles )

        this_bin = vrad_min
        vrad_bins.append(this_bin)
        this_label = str(this_bin)
        vrad_labels.append(this_label)

    vrad_bins.reverse() 
    vrad_labels.reverse() 

    r_min0 = 300
    r_step = 100

    for i in range(0, 12):
        r_min = r_min0 + i * r_step
        r_max = r_min + r_step

        data_new = filter_data(data=data, vrad_max=0.0, vrad_min=-200, r_max=r_max, r_min=r_min, vel_norm=vel_norm, mass_min=mass_min, vtan_max=vtan_max)

        vtan_bins = [0, 100, 1000]
        vtan_labels = ['Vtan<100', 'Vtan>100']

        vrad_binned = pd.cut(data_new['Vrad'], labels=vrad_labels, bins=vrad_bins)
        data_new.insert(1,'Vrad_bin',vrad_binned)

        vtan_binned = pd.cut(data_new['Vtan'], labels=vtan_labels, bins=vtan_bins)
        data_new.insert(1,'Vtan_bin',vtan_binned)

        sns.violinplot(y='Mlog', x='Vrad_bin', data=data_new, hue='Vtan_bin', split=True, inner="quartile")

        mlog_med = '%.3f' % np.median(data_new['Mlog'])

        slopes = np.polyfit(-data_new['Vrad_norm'], data_new['Mlog'], 1)

        slope = '%.3f' % slopes[0]
        title = 'R = [' + str(r_min) + ', ' + str(r_max) + '], median log10(Mtot) = ' + mlog_med + ', slope: ' + slope 
        out_file = 'output/violinplot_vrad_Mlog_R' + str(i) + '.png'

        print('Saving violin plot: ', out_file, ' sample size: ', len(data_new))

        plt.title(title)
        plt.savefig(out_file)
        plt.cla()
        plt.clf()
 

def distributions_1D(data_lgf=None, data_ahf=None, data_rs=None, col='Mlog', xlim=False):

    if xlim == True:
        plt.xlim(0, 500.0)
        plt.xlim(-500.0, 190.0)

    sns.distplot(data_lgf[col], color='blue', label='LGF')
    sns.distplot(data_ahf[col], color='red', label='SmallBox')
    sns.distplot(data_rs[col], color='green', label='BigMD')
    plt.legend()

    file_out = 'output/distplot_compare_' + col + '.png'
    plt.savefig(file_out)

    plt.clf()
    plt.cla()



'''
        MAIN PROGRAM
'''

#use_simu = 'LGF'
#use_simu = 'AHF'
use_simu = 'RS'

sns.set_style('whitegrid')

# Put some threshold on LG properties from different datasets
ratio_max = 4.0
mass_min = 6.0e+11
vrad_max = -10
vrad_min = -200.0
vtan_max = 1000.0
r_max = 1500.0
r_min = 300.0

# Normalization constants
mass_norm = 1.0e+12
vel_norm = 100.0
r_norm = 1000.0

# Apply some restriction on data properties
#filterData = True
filterData = False

# Read the different datasetsm rescale and filter

if use_simu == 'RS':
    data = rf.read_lg_rs_fullbox(TA=True)
elif use_simu == 'LGF': 
    data = rf.read_lg_lgf(TA=True)
elif use_simu == 'AHF':
    data = rf.read_lg_fullbox(TA=True)

data = filter_data(data=data, vrad_max=vrad_max, r_max=r_max, r_min=r_min, vel_norm=vel_norm, mass_min=mass_min, vtan_max=vtan_max)
data = rescale_data(data=data, mass_norm=mass_norm, r_norm=r_norm)

radial_velocity_binning(data=data)




'''
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
'''





