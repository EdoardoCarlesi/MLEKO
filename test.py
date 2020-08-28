import pandas as pd
import read_files as rf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


import mc_datagen as mc

mc.montecarlo(distribution='gauss')


'''
data = rf.read_lg_fullbox(TA=True); name_add = '_ahf'

vrad_max = -1.0
mass_norm = 1.0e+12
data = data[data['Vrad'] < vrad_max]

data['Mtot'] = data['M_M31'] + data['M_MW']
data['Mratio'] = data['M_M31'] / data['M_MW']
data['Mlog'] = np.log10(data['Mtot']/mass_norm)
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
'''
plt.show()
