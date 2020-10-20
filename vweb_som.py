'''
    CluesML
    Machine Learning tools for the CLUES project

    (C) Edoardo Carlesi 2020
    https://github.com/EdoardoCarlesi/CluesML
'''

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tools as t


"""
    MAIN PROGRAM STARTS
"""

plot3d = False
plotKLV = True
plotEVs = True
plotLambdas = True

file_base = '/home/edoardo/CLUES/DATA/Vweb/512/CSV/'
#web_file = 'vweb_00_10.000032.Vweb-csv'; str_grid = '_grid32'; grid = 32
#web_file = 'vweb_00_10.000064.Vweb-csv'; str_grid = '_grid64'; grid = 64
web_file = 'vweb_00_10.000128.Vweb-csv'; str_grid = '_grid128'; grid = 128

#web_file = 'vweb_128_.000128.Vweb-csv'; str_grid = '_box500grid128'; grid = 128
#web_file = 'vweb_256_.000256.Vweb-csv'; str_grid = '_box500grid256'; grid = 256

#box = 500.0; thick = 5.0
#box = 500.0e+3; thick = 5.0e+3
box = 100.0; thick = 2.0

web_df = pd.read_csv(file_base + web_file)
web_df['logdens'] = np.log10(web_df['dens'])

cols_select = ['l1', 'l2', 'l3']; vers = ''; str_som = r'SOM $\lambda$s'

X = web_df[cols_select].values

# Feature scaling
#sc = MinMaxScaler(feature_range = (0, 1))
#X = sc.fit_transform(X)
#print(X)

# Generate and train the actual Self Organizing Map
n_feat = len(cols_select)
n_x = 1; n_y = 6
n_clusters = n_x * n_y

print('Generating a ', n_x, 'x', n_y, ' SOM with ', n_feat, ' features.')

som = MiniSom(x = n_x, y  = n_y, input_len = n_feat, sigma = 0.3, learning_rate = 0.5, random_seed = 45, neighborhood_function='gaussian')
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 50, verbose = True)

web_type = np.zeros((len(web_df)), dtype=int)

for i, x in enumerate(X):
    web_type[i] = int(som.winner(x)[1])

#winner_x = np.array([som.winner(x) for x in X]).T
#print(winner_x[1])
web_df['env'] = web_type
#web_df['env'] = web_df[cols_select].apply(lambda x: som.winner(x))
#print(web_df.head(100))

out_evs_3d = 'output/som_3d_' + vers
out_evs_dist = 'output/som_vweb_' + vers
out_dens_dist = 'output/som_dens_' + vers
out_web_slice = 'output/som_web_lv_' + vers

if n_clusters == 2:
    colors = ['lightgrey', 'black']
    envirs = ['void', 'knot']
elif n_clusters == 3:
    colors = ['lightgrey', 'darkgrey', 'red']
    envirs = ['underdense', 'filament', 'knot']
elif n_clusters == 4:
    colors = ['lightgrey', 'grey', 'black', 'red']
    envirs = ['void', 'sheet', 'filament', 'knot']
elif n_clusters == 5:
    colors = ['lightgrey', 'grey', 'darkgrey', 'black', 'red']
    envirs = ['void', 'sheet', 'wall', 'filament', 'knot']
elif n_clusters == 6:
    colors = ['lightgrey', 'grey', 'darkgrey', 'black', 'orange', 'red']
    envirs = ['void', 'sheet', 'wall', 'filament', 'clump', 'knot']

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

    ind_vals = web_type #kmeans.labels_[web_df.index]

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
    plt.title('SOM $ n=$' + str(n_clusters), fontsize=fontsize)
    f_out = out_web_slice + str_grid + '.png'
    plt.tight_layout()
    plt.savefig(f_out)


