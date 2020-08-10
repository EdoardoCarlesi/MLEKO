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

data_rs = rf.read_lg_rs_fullbox()
data_rs['Mtot'] = data_rs['M_M31'] + data_rs['M_MW']
data_rs['Mratio'] = data_rs['M_M31'] / data_rs['M_MW']
data_rs['Mlog'] = np.log10(data_rs['Mtot'])
data_rs['Vtot'] = np.sqrt(data_rs['Vrad'] **2 + data_rs['Vtan'] **2)
data_rs['Vlog'] = np.log10(data_rs['Vtot'])

data_ahf = rf.read_lg_fullbox()
data_ahf['Mtot'] = data_ahf['M_M31'] + data_ahf['M_MW']
data_ahf['Mratio'] = data_ahf['M_M31'] / data_ahf['M_MW']
data_ahf['Mlog'] = np.log10(data_ahf['Mtot'])
data_ahf['Vtot'] = np.sqrt(data_ahf['Vrad'] **2 + data_ahf['Vtan'] **2)
data_ahf['Vlog'] = np.log10(data_ahf['Vtot'])

cols = ['R','Vrad', 'Vtan', 'Mtot']

#col = 'R'
col = 'Mlog'
#col = 'Mratio'
#col = 'Vlog'

sns.distplot(data_ahf[col])
sns.distplot(data_rs[col])
plt.show()


