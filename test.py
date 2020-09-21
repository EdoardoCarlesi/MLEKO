import pandas as pd
import read_files as rf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import data_visualization as dv
import montecarlo as mc

#mc.montecarlo(distribution='gauss')

vrad=[100, 220]
vtan=[1, 150]
rad=[450, 650]

cols=['Vrad', 'R', 'Vtan']
mc_df = mc.gen_mc(cols=cols, vrad=vrad, rad=rad, vtan=vtan, n_pts=1000)

#regressor0 = 'output/regressor_linear_mass_total_rs_model.pkl'
#regressor1 = 'output/regressor_linear_mass_ratio_rs_model.pkl'

regressor0 = 'output/regressor_gradient_boost_mass_mw_rs_model.pkl'
regressor1 = 'output/regressor_gradient_boost_mass_m31_rs_model.pkl'

mc.plot_mc_double(
                    show=True,
                    extra_info='mw_m31', 
                    regressor_type='linear',
                    regressor_file0=regressor0,
                    regressor_file1=regressor1,
                    cols=cols,
                    mc_df=mc_df,
                    mass_type='mass'
                    )


#def plot_mc_double(extra_info='mass_total', distribution='gauss', show=False, n_pts=1000, n_bins=15,
#        regressor_type='random_forest', regressor_file0=None, regressor_file1=None, cols=None, mc_df=None):



'''
data = rf.read_lg_fullbox(TA=True); name_add = '_ahf'

mass_norm = 1.0e+12
data['Mtot'] = data['M_M31'] + data['M_MW']
data['Mlog'] = np.log10(data['Mtot']/mass_norm)

print('Original data: ', len(data))

n_bins = 6

data_new = dv.equal_number_per_bin(data=data, bin_col='Mlog', n_bins=n_bins)

print(data_new.head())

vrad_max = -1.0
data = data[data['Vrad'] < vrad_max]

data['Mratio'] = data['M_M31'] / data['M_MW']
data['Mlog_TA'] = np.log10(data['M_TA'] / mass_norm)
data['Mratio_TA'] = np.log10(data['M_TA'] / data['Mtot'])

ta_median = np.median(data['Mratio_TA'])

print('Median: ', ta_median) 

sns.set_style('whitegrid')
plt.xlim([-2.0, 3.0])
sns.distplot(data['Mratio_TA'], bins=50)
title = 'TA to true M ratio median = ' + '%.3f' % ta_median
plt.title(title)
plt.savefig('output/ratio_MTA_true')
plt.show()
'''
