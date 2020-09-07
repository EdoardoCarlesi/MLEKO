import pandas as pd
import read_files as rf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import data_visualization as dv
import montecarlo as mc


def plot_1d_mw_m31():

    data = rf.read_lg_fullbox(TA=False); name_add = '_ahf'

    mass_norm = 1.0e+12
    vrad_max = -1.0
    mratio_max = 5.0

    data['Mratio'] = data['M_M31'] / data['M_MW']

    data = data[data['Vrad'] < vrad_max]
    data = data[data['Mratio'] < mratio_max]

    data['M_M31'] = np.log10(data['M_M31']/mass_norm)
    data['M_MW'] = np.log10(data['M_MW']/mass_norm)

    dv.distribution_1D(data=data)



def plot_mc_masses():
    vrad=[1, 220]
    vtan=[1, 150]
    rad=[450, 1050]

    cols=['Vrad', 'R', 'Vtan']
    
    mc_df = mc.gen_mc(cols=cols, vrad=vrad, rad=rad, vtan=vtan, n_pts=10000)

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




def plot_timing_argument():
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

    plt.xlim([-2.0, 3.0])
    sns.distplot(data['Mratio_TA'], bins=50)
    title = 'TA to true M ratio median = ' + '%.3f' % ta_median
    plt.title(title)
    plt.savefig('output/ratio_MTA_true')
    plt.show()



'''
    MAIN PROGRAM 
'''

plot_1d_mw_m31()

plot_mc_masses()






