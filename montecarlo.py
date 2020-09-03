import pickle
import numpy as np
import pandas as pd
import random as rd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor


def gen_mc(distribution='gauss', cols=None, n_pts=1000, sanity_check=False, **kwargs):
    # Generate a random list of integers
    rd.seed()

    # Put everything into a dataframe
    mc_df = pd.DataFrame()

    i_col = 0
    for arg in kwargs.values():

        if distribution == 'flat':
            int_list = rd.sample(range(0, n_pts), n_pts)

            d = (arg[1] - arg[0]) / float(n_pts)

            # Initialize some arrays
            values = np.zeros((n_pts))

            # Fill the arrays
            for i, step in enumerate(int_list):
                values[i] = (arg[0] + step * d)

        elif distribution == 'gauss':
            med = np.median(arg)
            sig = np.abs(med - arg[0])

            values = np.random.normal(loc=med, scale=sig, size=n_pts)

        mc_df[cols[i_col]] = values
        i_col = i_col + 1

    if sanity_check == True:
        print(mc_df.head())
    
    return mc_df


def plot_mc_simple(extra_info='mass_total', distribution='gauss', show=False, n_pts=1000, n_bins=15, 
        regressor_type='random_forest', regressor_file=None, cols=None, mc_df=None):

    regressor = pickle.load(open(regressor_file, 'rb'))
    print('Loading regression model from: ', regressor_file)

    mc_df = mc_df.dropna()
    X_mc = mc_df[cols]
    predict_mc = regressor.predict(X_mc)

    percs = np.percentile(predict_mc, [25, 50, 75])
    print('Percentiles: ', percs) 
    
    title_perc = '%.3f %.3f %.3f' % (percs[0], percs[1], percs[2])

    sns.distplot(predict_mc, bins=n_bins)
    plt.xlabel('log10(Mtot)')
    title = 'MC Mtot ' + regressor_type + ' pct ' + title_perc
    plt.title(title)
    out_name = 'output/montecarlo_' + regressor_type + extra_info + '.png'
    plt.tight_layout()
    plt.savefig(out_name)

    if show == True:
        plt.show(block=False)
        plt.pause(4)
        plt.close()


def plot_mc_double(extra_info='mass_total', distribution='gauss', show=False, n_pts=1000, n_bins=15, 
        regressor_type='random_forest', regressor_file0=None, regressor_file1=None, cols=None, mc_df=None):

    regressor0 = pickle.load(open(regressor_file0, 'rb'))
    print('Loading regression model from: ', regressor_file0)

    regressor1 = pickle.load(open(regressor_file1, 'rb'))
    print('Loading regression model from: ', regressor_file1)

    mc_df = mc_df.dropna()
    X_mc = mc_df[cols]
    
    # Total mass
    mtot = regressor0.predict(X_mc)
 
    # Mass ratio
    ratio = regressor1.predict(X_mc)
   
    # MW mass 
    n0 = len(predict_mc0)
    mmw = np.zeros((n0))
    mm31 = np.zeros((n0))
    
    for i, mt in enumerate(mtot):
        mmw[i] = mt / (1.0 + ratio[i])
        #mmw[i] = 10 ** mmw[i]
        mm31[i] = mmw[i] * ratio[i]

    title_perc = '%.3f %.3f' % (perc_mw[1], perc_m31[1]) 

    sns.distplot(mmw, bins=n_bins, color='blue')
    sns.distplot(mm31, bins=n_bins, color='red')
    #plt.xlabel('log10(Mtot)')
    title = 'MC Mtot ' + regressor_type + ' pct ' + title_perc
    plt.title(title)
    out_name = 'output/montecarlo_' + regressor_type + extra_info + '.png'
    plt.tight_layout()
    plt.savefig(out_name)

    if show == True:
        plt.show(block=False)
        plt.pause(4)
        plt.close()



