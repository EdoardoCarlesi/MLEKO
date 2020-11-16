'''
    CluesML
    Machine Learning tools for the CLUES project

    (C) Edoardo Carlesi 2020
    https://github.com/EdoardoCarlesi/CluesML
'''

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats

import pandas as pd
import numpy as np
import glob, os
import cv2

def distance(v0, v1):

    v = (v0 - v1)**2
    return np.sqrt(np.sum(v))


'''
    Find the percentiles and error bars
'''
def percentiles(x=None, perc=[20, 50, 80]):

    pct = np.percentile(x, perc)

    print('Percentiles: ', pct)
    print('%.3f_{%.3f}^{%.3f}' % (pct[1], pct[0]-pct[1], pct[2]-pct[1]))

    return pct


'''
    Do a PCA analysis of a dataset
'''
def data_pca(data=None, columns=None, pca_percent=None):

    print('Doing PCA reduction of dataset: ', columns)

    # Initialize PCA
    if pca_percent == None:
        n_components = len(columns)
        pca = PCA(n_components = n_components)

    else:
        pca = PCA(pca_percent)

    # Select these data from the full dataframe
    X = data[columns]

    # Normalize the data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Transform
    principal_components = pca.fit_transform(X)

    if pca_percent != None:
        n_components = principal_components.shape[1]

    # Rename the feature columns in the new basis
    feat_cols = ['feature: ' + str(i) for i in range(0, n_components)]

    # Put the PCA transformed data into a DataFrame
    pc_df = pd.DataFrame(data=principal_components, columns=feat_cols)

    # Plot some information
    print('PCA explained variance ratio is: ', pca.explained_variance_ratio_)

    return pc_df

'''
    Check the shape of an input image
'''
def check_image(img):
    f = cv2.imread(img)
    print(img,  ' has shape: ', f.shape)

    return f.shape

'''
    Spit out the path to all the files within a given folder
'''
def find_images_in_folder(path=None, extension='png'):
    os.chdir(path)
    img_list = []

    for img in glob.glob('*.'+extension):
        img_list.append(path + img)

    return img_list

'''
    Mean absolute percentage error
'''
def MAPE(y_true, y_pred):
    d_y = np.abs(y_true - y_pred)
    d_y = np.abs((d_y) / (y_true))

    return d_y


# Normalize manually - do not rely on sklearn auto-matic methods
def normalize_data(data=None, mode='boxcox', verbose=False):

    mass_norm = 1.0e+12
    vel_norm = 100.0
    r_norm = 1000.0

    if verbose == True:
        print('Skew R   : ', data['R'].skew())
        print('Skew Vrad: ', data['Vrad'].skew())
        print('Skew Vtan: ', data['Vtan'].skew())
        print('Skew Mtot: ', data['Mtot'].skew())
        print('Skew MW  : ', data['M_MW'].skew())
        print('Skew M31 : ', data['M_M31'].skew())

    data['R'] = data['R'] / r_norm

    if mode == 'log':
        data['Vrad'] = np.log10(-data['Vrad']/vel_norm)
        data['Vtan'] = np.log10( data['Vtan']/vel_norm)
        data['Mtot'] = np.log10(data['Mtot']/mass_norm)
        data['M_MW'] = np.log10(data['M_MW']/mass_norm)
        data['M_M31'] = np.log10(data['M_M31']/ mass_norm)

    elif mode == 'boxcox':
        data['Vrad'] = stats.boxcox(-data['Vrad'])[0]
        data['Vtan'] = stats.boxcox(data['Vtan'])[0]
        data['Mtot'] = stats.boxcox(data['Mtot'])[0]
        data['M_MW'] = stats.boxcox(data['M_MW'])[0]
        data['M_M31'] = stats.boxcox(data['M_M31'])[0]

    elif mode == 'mix1':
        data['Vrad'] = stats.boxcox(-data['Vrad'])[0]
        data['Vtan'] = stats.boxcox(data['Vtan'])[0]
        data['Mtot'] = np.log10(data['Mtot']/mass_norm)
        data['M_MW'] = np.log10(data['M_MW']/mass_norm)
        data['M_M31'] = np.log10(data['M_M31']/mass_norm)

    elif mode == 'lin':
        vel_norm = 500.0
        data['Vrad'] = -data['Vrad']/vel_norm
        data['Vtan'] = data['Vtan']/vel_norm
        data['Mtot'] = data['Mtot']/mass_norm
        data['M_MW'] = data['M_MW']/mass_norm
        data['M_M31'] = data['M_M31']/mass_norm

    elif mode == 'mix2':
        vel_norm = 1000.0
        data['Vrad'] = np.log10(-data['Vrad']/vel_norm)
        data['Vtan'] = np.log10( data['Vtan']/vel_norm)
        data['Mtot'] = data['Mtot']/mass_norm
        #data['Mtot'] = np.log10(data['Mtot'])
        #data['M_MW'] = np.log10(data['M_MW'])
        #data['M_M31'] = np.log10(data['M_M31'])
        #data['Mtot'] = stats.boxcox(data['Mtot'])[0]
        #data['M_MW'] = stats.boxcox(data['M_MW'])[0]
        #data['M_M31'] = stats.boxcox(data['M_M31'])[0]


    if verbose == True:
        print('----')
        print('Skew R   : ', data['R'].skew())
        print('Skew Vrad: ', data['Vrad'].skew())
        print('Skew Vtan: ', data['Vtan'].skew())
        print('Skew Mtot: ', data['Mtot'].skew())
        print('Skew MW  : ', data['M_MW'].skew())
        print('Skew M31 : ', data['M_M31'].skew())

    return data



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



def filter_data(data=None, vrad_max=-1.0, vrad_min=-200.0, r_max=1500.0, r_min=350.0, mass_min=5.0e+11, vel_norm=100.0, vtan_max=500.0, ratio_max=5.0):

    data = data[data['Vrad'] < vrad_max]
    data = data[data['Vrad'] > vrad_min]
    data = data[data['R'] < r_max]
    data = data[data['R'] > r_min]
    data = data[data['Vtan'] < vtan_max]
    data = data[data['M_MW'] > mass_min]
    data = data[data['Mratio'] < ratio_max]

    return data



def equal_number_per_bin(data=None, n_bins=10, bin_col='Mlog'):
    
    bins = np.zeros((n_bins))

    bins[0] = data[bin_col].min()
    bins[n_bins-1] = data[bin_col].max()
    bin_size = (bins[n_bins-1] - bins[0]) / float(n_bins)
 
    for i in range(1, n_bins-1):
        bins[i] = bin_size * i

    new_data = data[data[bin_col] > bins[n_bins-2]]
    n_sample = len(new_data)

    for i in range(1, n_bins-1):
        this_data = data[data[bin_col] < bins[n_bins-i-1]] 
        this_data = this_data[this_data[bin_col] > bins[n_bins-i-2]]

        if len(this_data) > n_sample:
            this_data = this_data.sample(n=n_sample)

        new_data = pd.concat([new_data, this_data])

    print('NewData: ', len(new_data))

    return new_data



def do_pca(data=None, pca_percent=0.9, pca_cols=['R', 'Vrad', 'Vtan']):
    # Do a PCA to check the data
    pca_cols = ['R','Vrad', 'Vtan'] #, 'AngMom', 'Energy'] #, l1, l2, l3, dens]
    data_pca = t.data_pca(data=data, columns=pca_cols, pca_percent=pca_percent)
    print('PCA at ', pca_percent, ' n_components: ', len(data_pca.columns), ' n_original: ', len(all_columns))
    print(data_pca.info())
    print(data_pca.head())

    return data_pca



def radial_velocity_binning(use_simu=None, data=None, vrad_max=-10.0, vrad_min=-120.0, 
                           r_max=1500.0, r_min=300.0, vel_norm=100.0, mass_min=4.0e+11, vtan_max=200.0):

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
 
