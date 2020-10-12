'''
    CluesML
    Machine Learning tools for the CLUES project

    (C) Edoardo Carlesi 2020
    https://github.com/EdoardoCarlesi/CluesML
'''

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tools as t


def plot_vweb(data=None, fout=None, thresh=0.0, grid=64):
    z_min = 48.0
    z_max = 52.0

    data = data[data['z'] > z_min]
    data = data[data['z'] < z_max]

    shift = 50.0
    data['x'] = data['x'].apply(lambda x: x - shift)
    data['y'] = data['y'].apply(lambda x: x - shift)
    
    voids = data[data['l1'] < thresh]
    sheet = data[(data['l2'] < thresh) & (data['l1'] > thresh)]
    filam = data[(data['l2'] > thresh) & (data['l3'] < thresh)]
    knots = data[data['l3'] > thresh]

    fontsize = 20

    if grid == 32:
        size = 40
    elif grid == 64:
        size = 15
    elif grid == 128:
        size = 5

    print('Plotting web with lambda threshold ')
    # Plot the eigenvaule threshold based V-Web
    plt.figure(figsize=(10, 10))
    plt.xlim([-shift, shift])
    plt.ylim([-shift, shift])
    plt.title('$\lambda_{thr} = $' + str(thresh), fontsize=fontsize)
    plt.xlabel(r'SGX $\quad [h^{-1} Mpc]$', fontsize=fontsize)
    plt.ylabel(r'SGY $\quad [h^{-1} Mpc]$', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.scatter(voids['x'], voids['y'], c='lightgrey', s=size, marker='s')
    plt.scatter(sheet['x'], sheet['y'], c='grey', s=size, marker='s')
    plt.scatter(filam['x'], filam['y'], c='black', s=size, marker='s')
    plt.scatter(knots['x'], knots['y'], c='red', s=size, marker='s')

    # Save file
    f_out = fout + '_' + str(thresh).replace('.','') + '.png'
    plt.tight_layout()
    print('Saving fig to ', f_out)
    plt.savefig(f_out)
    plt.cla()
    plt.clf()

    # Plot densities
    palette="YlOrBr"
    plt.figure(figsize=(10, 10))
    plt.xlim([-shift, shift])
    plt.ylim([-shift, shift])
    plt.title('$\log_{10}\Delta_m, k =$' + str(n_clusters), fontsize=fontsize)
    sns.scatterplot(data['x'], data['y'], hue=np.log10(data['dens']), marker='s', s=size*3, legend = False, palette=palette)

    # Override seaborn defaults
    plt.xlabel(r'SGX $\quad [h^{-1} Mpc]$', fontsize=fontsize)
    plt.ylabel(r'SGY $\quad [h^{-1} Mpc]$', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()

    # Save file
    f_out = fout + '_dens.png'
    plt.savefig(f_out)
    plt.cla()
    plt.clf()


   
"""
    MAIN PROGRAM STARTS
"""

plotStd = False
plot3d = True
plotKLV = False
plotEVs = False
plotLambdas = False

file_base = '/home/edoardo/CLUES/DATA/Vweb/512/CSV/'
#web_file = 'vweb_00_10.000032.Vweb-csv'; str_grid = '_grid32'; grid = 32
#web_file = 'vweb_00_10.000064.Vweb-csv'; str_grid = '_grid64'; grid = 64
web_file = 'vweb_00_10.000128.Vweb-csv'; str_grid = '_grid128'; grid = 128

#web_file = 'vweb_25_15.000064.Vweb-csv'; str_grid = '_grid64'
#web_file = 'vweb_01_10.000064.Vweb-csv'

web_df = pd.read_csv(file_base + web_file)

if plotStd == True: 

    f_out = 'output/kmeans_oldvweb' + str_grid
    for thresh in [0.0, 0.1, 0.2]:
        plot_vweb(fout=f_out, data=web_df, thresh=thresh, grid=grid)

web_df['logdens'] = np.log10(web_df['dens'])
#print(web_df.head())

cols_select = ['l1', 'l2', 'l3']; vers = ''; str_kmeans = r'$k$-means $\lambda$s'
#cols_select = ['l1', 'l2', 'l3', 'dens']; vers = 'd'; str_kmeans = r'$k$-means $\lambda$s, \Delta_m'
#cols_select = ['l1', 'l2', 'l3', 'logdens']; vers = 'ld'; str_kmeans = r'$k$-means $\lambda$s, $\log_{10}\Delta_m$'
#cols_select = ['l1', 'l2', 'l3', 'dens', 'Vx', 'Vy', 'Vz']; vers = 'd_vx'

web_ev_df = web_df[cols_select]

n_clusters = 3

kmeans = KMeans(n_clusters = n_clusters, n_init = 20)
kmeans.fit(web_ev_df)

centers = kmeans.cluster_centers_

#print(type(kmeans.labels_))
web_df['env'] = kmeans.labels_
#ntot = len(web_df)

if n_clusters == 4:
    colors = ['lightgrey', 'grey', 'black', 'red']
    envirs = ['void', 'sheet', 'filament', 'knot']
elif n_clusters == 3:
    colors = ['lightgrey', 'darkgrey', 'red']
    envirs = ['underdense', 'filament', 'knot']
elif n_clusters == 5:
    colors = ['lightgrey', 'grey', 'darkgrey', 'black', 'red']
    envirs = ['void', 'sheet', 'wall', 'filament', 'knot']

vers = vers + '_k' + str(n_clusters)
out_evs_3d = 'output/kmeans_3d_' + vers
out_evs_dist = 'output/kmeans_vweb_' + vers
out_dens_dist = 'output/kmeans_dens_' + vers
out_web_slice = 'output/kmeans_web_lv_' + vers

envirs_sort = []
colors_sort = []

deltas = np.zeros((n_clusters))
ntot = len(web_df)

for i in range(0, n_clusters):
    deltas[i] = np.median(web_df[web_df['env'] == i]['dens'].values)

deltas_sort = np.sort(deltas)

for i in range(0, n_clusters):
    n = len(web_df[web_df['env'] == i]) 
    index = np.where(deltas_sort == deltas[i])
    env_str = envirs[index[0][0]]
    envirs_sort.append(env_str)
    colors_sort.append(colors[index[0][0]])

    num_str = '& $ %.3f $ & $%.3f$ ' % (deltas[i], n/ntot)
    tab_str = env_str + num_str
    print(tab_str)

print('\n')
for i in range(0, n_clusters):
    c_str = ' & $ %.3f $  &  $ %.3f $ & $ %.3f $ ' % (centers[i,0], centers[i, 1], centers[i, 2])
    line_str = envirs_sort[i] + c_str
    print(line_str)

cols = []

if plot3d == True:
    for il, cl in enumerate(kmeans.labels_):
        cols.append(colors_sort[cl])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(web_ev_df['l1'], web_ev_df['l2'], web_ev_df['l3'], c = cols)
    ax.set_xlabel(r'$\lambda_1$')
    ax.set_ylabel(r'$\lambda_2$')
    ax.set_zlabel(r'$\lambda_3$')
    plt.tight_layout()
    f_out = out_evs_3d + str_grid + '.png'
    plt.savefig(f_out)
    #plt.show(block = False)
    #plt.pause(3)
    #plt.close()

if plotEVs == True:
    maxrange = n_clusters
else:
    maxrange = 0

for i in range(0, maxrange):
    evs_df = web_df[web_df['env'] == i]

    # Only plot the first three axes i.e. the eigenvalues
    for col in cols_select[0:3]:
        sns.distplot(evs_df[col])
    
    env_str = envirs_sort[i]
    f_out = out_evs_dist + env_str + str_grid + '.png'

    print('Plotting l to: ', f_out)
    fontsize=10

    # Plot the three eigenvalues
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(envirs_sort[i] + ' $k=$' + str(n_clusters), fontsize=fontsize)
    plt.xlabel(r'$\lambda_3, \lambda_2, \lambda_2$', fontsize=fontsize)

    if grid == 32:
        plt.xlim(-0.5, 0.5)
    elif grid == 128:
        plt.xlim(-1.5, 3.0)

    plt.savefig(f_out)
    plt.clf()
    plt.cla()

    # Now plot the delta distribution
    if grid == 32:
        plt.xlim(-1, 1)
    elif grid == 128:
        plt.xlim(-2, 2)

    nbins = 100
    #plt.xscale('log')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    deltam = '%.3f' % deltas[i]
    plt.title(env_str + r': $ \bar \Delta_m = $' + deltam + ' $, k=$' + str(n_clusters))
    sns.distplot(np.log10(evs_df['dens']), bins=nbins)

    plt.xlabel(r'$\log_{10}\Delta _m$', fontsize=fontsize)
    f_out = out_dens_dist + env_str + str_grid + '.png'
    print('Plotting d to: ', f_out)
    plt.savefig(f_out)
    plt.clf()
    plt.cla()


if plotLambdas == True:

    labels = ['$\lambda_1$', '$\lambda_2$', '$\lambda_3$']
    web_df['env_name'] = web_df['env'].apply(lambda x: envirs_sort[x])

    for il, col in enumerate(cols_select):

        for ie, env in enumerate(envirs):
            tmp_env = web_df[web_df['env_name'] == env]
            sns.distplot(tmp_env[col], color=colors[ie], label=env)

        env_str = envirs_sort[i]
        f_out = out_evs_dist + str_grid + col + '.png'

        print('Plotting l to: ', f_out)
        fontsize=10

        # Plot the three eigenvalues
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        if grid == 32:
            plt.xlim([-0.5, 0.5])
        elif grid == 128:
            plt.xlim([-1, 3.0])

        plt.xlabel(labels[il], fontsize=fontsize)
        plt.legend()
        plt.tight_layout()
    
        # Save figure and clean plot
        plt.savefig(f_out)
        plt.clf()
        plt.cla()


# Plot a slice of the local volume
if plotKLV == True:
    z_min = 48.00
    z_max = 52.00

    web_df = web_df[web_df['z'] > z_min]
    web_df = web_df[web_df['z'] < z_max]

    web_df['env_name'] = web_df['env'].apply(lambda x: envirs_sort[x])

    ind_vals = kmeans.labels_[web_df.index]

    web_df['x'] = web_df['x'].apply(lambda x: x - 50.0)
    web_df['y'] = web_df['y'].apply(lambda x: x - 50.0)

    lim = 50

    fontsize = 20

    if grid == 32:
        size = 100
    elif grid == 64:
        size = 20
    elif grid == 128:
        size = 5
        
    plt.figure(figsize=(10, 10))
    plt.xlim([-lim, lim])
    plt.ylim([-lim, lim])

    for ie, env in enumerate(envirs):
        tmp_env = web_df[web_df['env_name'] == env]
        plt.scatter(tmp_env['x'], tmp_env['y'], c=colors[ie], s=size, marker='s')

    plt.xlabel(r'SGX $\quad [h^{-1} Mpc]$', fontsize=fontsize)
    plt.ylabel(r'SGY $\quad [h^{-1} Mpc]$', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(str_kmeans + ' $k=$' + str(n_clusters), fontsize=fontsize)
    f_out = out_web_slice + str_grid + '.png'
    plt.tight_layout()
    plt.savefig(f_out)



