import pickle
import numpy as np
import pandas as pd
import random as rd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor


def montecarlo(vrad=[100.0, 120.0], vtan=[1.0, 160.0], rad=[600.0, 700.0], extra_info='mass_total', distribution='flat', show=False,
                n_pts=1000, n_bins=15, regressor_type='random_forest', regressor_file=None, cols=['Vrad_norm', 'R', 'Vtan_norm']):

    # Generate a random list of integers
    rd.seed()

    if distribution == 'flat':
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

    elif distribution == 'gauss':
        vrad_med = np.median(vrad)
        vrad_sig = np.abs(vrad_med - vrad[0])

        vtan_med = np.median(vtan)
        vtan_sig = np.abs(vtan_med - vtan[0])

        rad_med = np.median(rad)
        rad_sig = np.abs(rad_med - rad[0])

        vrads = np.random.normal(loc=vrad_med, scale=vrad_sig, size=n_pts)
        vtans = np.random.normal(loc=vtan_med, scale=vtan_sig, size=n_pts)
        rads = np.random.normal(loc=rad_med, scale=rad_sig, size=n_pts)

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

    regressor = pickle.load(open(regressor_file, 'rb'))
    print('Loading regression model from: ', regressor_file)

    mc_df = mc_df.dropna()
    X_mc = mc_df[cols]
    predict_mc = regressor.predict(X_mc)

    print('Percentiles: ', np.percentile(predict_mc, [25, 50, 75]))

    sns.distplot(predict_mc, bins=n_bins)
    plt.xlabel('log10(Mtot)')
    title = 'MC Mtot ' + regressor_type  
    plt.title(title)
    out_name = 'output/montecarlo_' + regressor_type + extra_info + '.png'
    plt.tight_layout()
    plt.savefig(out_name)

    if show == True:
        plt.show()

