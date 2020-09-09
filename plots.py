import pandas as pd
import read_files as rf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import data_visualization as dv
import montecarlo as mc


def plot_1d_mw_m31():

    data = rf.read_lg_fullbox(TA=False); name_add = '_ahf'
    #data = rf.read_lg_rs_fullbox(TA=False); name_add = '_rs'
    #data = rf.read_lg_lgf(TA=False); name_add = '_lgf'

    mass_norm = 1.0e+12
    vrad_max = -1.0
    mratio_max = 5.0

    data['Mratio'] = data['M_M31'] / data['M_MW']

    data = data[data['Vrad'] < vrad_max]
    data = data[data['Mratio'] < mratio_max]

    data['M_M31'] = np.log10(data['M_M31']/mass_norm)
    data['M_MW'] = np.log10(data['M_MW']/mass_norm)

    dv.mass_distribution_1D(data=data, name_add=name_add, xlim=2.5)




def plot_mc_masses():
    vrad=np.log10([1.00, 1.20])
    vtan=np.log10([0.01, 1.50])
    rad=[450, 550]

    cols=['Vrad', 'R', 'Vtan']
    
    mc_df = mc.gen_mc(cols=cols, vrad=vrad, rad=rad, vtan=vtan, n_pts=10000)

    #regressor0 = 'output/regressor_linear_mass_total_rs_model.pkl'
    #regressor1 = 'output/regressor_linear_mass_ratio_rs_model.pkl'

    regressor0 = 'output/regressor_gradient_boost_mass_mw_sb_model.pkl'
    regressor1 = 'output/regressor_gradient_boost_mass_m31_sb_model.pkl'

    mc.plot_mc_double(
                    show=True,
                    extra_info='mw_m31', 
                    regressor_type='gradient_boost',
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

    vrad_max = -1.0
    data = data[data['Vrad'] < vrad_max]

    data['Mratio'] = data['M_M31'] / data['M_MW']
    data['Mlog_TA'] = np.log10(data['M_TA'] / mass_norm)
    data['Mratio_TA'] = np.log10(data['M_TA'] / data['Mtot'])
    #data['Mratio_TA'] = data['M_TA'] / data['Mtot']

    ta_median = np.median(data['Mratio_TA'])
    pct = np.percentile(data['Mratio_TA'], [20, 50, 80])

    print('Median: ', ta_median) 

    plt.figure(figsize=(5, 5))
    plt.xlim([-2.0, 3.0])
    sns.distplot(data['Mratio_TA'], bins=50)
    title = 'TA to true M ratio median = %.3f %.3f %.3f' % (pct[0], pct[1], pct[2])
    plt.title(title)
    plt.savefig('output/MTA_ratio')
    plt.cla()
    plt.clf()

    slope = np.polyfit(data['Mlog_TA'], data['Mlog'], 1)
    title = 'TA vs true M slope: ' + '%.3f' % slope[0]

    yy = []
    x = [data['Mlog_TA'].min(), data['Mlog_TA'].max()]

    for xx in x:
        yy.append(slope[0] * xx + slope[1])

    sns.kdeplot(data['Mlog_TA'], data['Mlog'])
    sns.scatterplot(data['Mlog_TA'], data['Mlog']) 
    sns.lineplot(x, x)
    sns.lineplot(x, yy)
    plt.title(title)

    plt.xlim([-1.0, 3.0])
    plt.savefig('output/MTA_kdeplot')
    plt.tight_layout()



'''
    MAIN PROGRAM 
'''

#plot_1d_mw_m31()
plot_mc_masses()
#plot_timing_argument()
#mc.plot_mc_simple()




