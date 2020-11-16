'''
    CluesML
    Machine Learning tools for the CLUES project

    (C) Edoardo Carlesi 2020
    https://github.com/EdoardoCarlesi/CluesML
'''

from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import type_metric, distance_metric

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, homogeneity_score, calinski_harabasz_score, silhouette_samples

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tools as t


def distance_(x, y, alpha):
    return np.power(np.sum((x - y)**2), alpha)
   
"""
    MAIN PROGRAM STARTS
"""

# Program settings
normalize = False

evalMetrics = True

plotNew = False
plotStd = False
plot3d = False
plotKLV = False
plotEVs = False
plotLambdas = False

file_base = '/home/edoardo/CLUES/DATA/Vweb/512/CSV/'
web_file = 'vweb_00_10.000032.Vweb-csv'; str_grid = '_grid32'; grid = 32
#web_file = 'vweb_00_10.000064.Vweb-csv'; str_grid = '_grid64'; grid = 64
#web_file = 'vweb_00_10.000128.Vweb-csv'; str_grid = '_grid128'; grid = 128


web_df = pd.read_csv(file_base + web_file)

n_centers = 4
n_pts = len(web_df)
resample_pts = int(n_pts / 10.0)

print(f'Resampling {resample_pts} points from a total of {n_pts}')

cols_select = ['l1', 'l2', 'l3']; vers = ''; str_kmeans = r'$k$-medioids $\lambda$s'
web_df = pd.read_csv(file_base + web_file)
web_select = web_df[cols_select].sample(resample_pts)
init_centers = web_select.sample(n_centers)

print(init_centers.head().values)

#alphas = [0.2, 0.33, 0.5, 0.66, 0.75]

alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

for alpha in alphas:
    distance = lambda x, y: np.power(np.sum((x-y)**2), alpha)
    metric = distance_metric(type_metric.USER_DEFINED, func = distance)
    km = kmeans(web_select, init_centers, metric = metric)
    km.process()

    clusters = km.get_clusters()

    for i, cluster in enumerate(clusters):
        csize = len(cluster)
        print(f'alpha = {alpha}, cluster ({i}) size = {csize}')


'''
box = 100.0; thick = 2.0


#plot_new_form(data=web_df, f_out='output/evs_new_form')

n_clusters = 10
n_init = 1

vers = vers + '_k' + str(n_clusters)
out_evs_new = 'output/kmeans_new_' + vers
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
    z_min = box * 0.5 - thick
    z_max = box * 0.5 + thick

    web_df = web_df[web_df['z'] > z_min]
    web_df = web_df[web_df['z'] < z_max]

    web_df['env_name'] = web_df['env'].apply(lambda x: envirs_sort[x])

    ind_vals = kmeans.labels_[web_df.index]

    web_df['x'] = web_df['x'].apply(lambda x: x - 50.0)
    web_df['y'] = web_df['y'].apply(lambda x: x - 50.0)

    lim = box * 0.5

    fontsize = 20

    if grid == 32:
        size = 100
    elif grid == 64:
        size = 20
    elif grid == 128:
        size = 5
    elif grid == 256:
        size = 2
        
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

'''

