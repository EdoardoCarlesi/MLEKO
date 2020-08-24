import pickle
import numpy as np
import pandas as pd
import random as rd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

# Multiply by -1 at the and
vrad = [1.0, 120]; vtan = [1.0, 1200]; rad  = [7.0, 1200]
#vrad = [100.0, 130.0]; vtan = [1.0, 160]; rad  = [600.0, 700.0]

n_pts=10000

# Generate a random list of integers
rd.seed()
int_list = rd.sample(range(0, n_pts), n_pts)

# Resolution
d_vrad = (vrad[1] - vrad[0]) / float(n_pts)
d_vtan = (vtan[1] - vtan[0]) / float(n_pts)
d_rad = (rad[1] - rad[0]) / float(n_pts)

# Initialize some arrays
vrads = np.zeros((n_pts))
vtans = np.zeros((n_pts))
rads = np.zeros((n_pts))

# Fill the arrays
for i, step in enumerate(int_list):
    vrads[i] = -(vrad[0] + step * d_vrad)
    vtans[i] =  (vtan[0] + step * d_vtan)
    rads[i] =  (rad[0] + step * d_rad)

# Put everything into a dataframe
mc_df = pd.DataFrame()
mc_df['R'] = rads
mc_df['Vrad'] = vrads
mc_df['Vtan'] = vtans
mc_df['R_norm'] = rads/1000.0
mc_df['Vrad_norm'] = np.log10(-vrads/100.0)
mc_df['Vtan_norm'] = np.log10(vtans/100.0)

# Sanity check
#print(mc_df.head())

# Now load a Ml model 
regressor_type = 'decision_tree'
#regressor_type = 'random_forest'
#regressor_type = 'gradient_boost'
#regressor_type = 'linear'
regressor_file = 'output/' + regressor_type + '_mass_total_model.pkl'

regressor = pickle.load(open(regressor_file, 'rb'))

print('Loaded ', regressor_file)

cols = ['Vrad_norm', 'R', 'Vtan_norm']

mc_df = mc_df.dropna()

X_mc = mc_df[cols]

predict_mc = regressor.predict(X_mc)

print('Percentiles: ', np.percentile(predict_mc, [25, 50, 75]))

sns.distplot(predict_mc, bins=20)
plt.xlabel('log10(Mtot)')
title = 'MC Mtot ' + regressor_type  
plt.title(title)
out_name = 'output/' + regressor_type + '_mc_mass_total_bigmd.png'
plt.tight_layout()
plt.savefig(out_name)
plt.show()



