'''
    CluesML
    Machine Learning tools for the CLUES project

    (C) Edoardo Carlesi 2020
    https://github.com/EdoardoCarlesi/CluesML
'''

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import silhouette_score, homogeneity_score, calinski_harabasz_score, silhouette_samples
from yellowbrick.cluster import KElbowVisualizer

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tools as t


def kmeans_stability(data=None, n_clusters_max=None, n_ks=10, rescale_factor=1000, verbose=False, f_out=None):
    
    # Rescale the dataset size for faster convergence
    n_data = int(len(data) / rescale_factor)
    data = data.sample(n_data, random_state = 1389)

    # This array will be returned at the end. It contains k, median error, median std on the error, median distance between clusters and median scatter over median distance
    results = np.zeros((n_clusters_max-1, 5))
    df_print = pd.DataFrame()

    columns = ['k', 'med_center', 'med_center_std', 'med_dist', 'med_center_med_dist']

    for n_clusters in range(2, n_clusters_max+1):
        centers = []
        minima = []

        # For each n_cluster value do n different iterations to check how much variance there is
        for i_k in range(0, n_ks):

            random_state = 10 * (i_k + n_clusters) + 1389

            if verbose:
                print(f'For n_clusters_max = {n_clusters}, iteration number = {i_k}, random state = {random_state}')

            kmeans = KMeans(n_clusters = n_clusters, n_init = 1, random_state = random_state)
            kmeans.fit(data)

            X = data.values
            labels = kmeans.fit_predict(X)
            centers.append(kmeans.cluster_centers_)
    
        minima = []
        all_dist = []

        # Look at the scatter between cluster centers
        for i_c, center in enumerate(centers):

            for i_t, this_center in enumerate(center):
                center_dist = []
                inter_dist = []

                for this_cluster in center[i_t+1:]:
                    inter = t.distance(this_center, this_cluster)
                    inter_dist.append(inter)

                siz_ic = len(inter_dist)

                if siz_ic == n_clusters-1:
                    avg_ic = np.median(inter_dist)
                    std_ic = np.std(inter_dist)

                    if verbose:
                        print(f'Average intracluster distance: {avg_ic}, stddev: {std_ic}, size: {siz_ic}')

                    # This is the average distance between centers for all the k means realizations
                    all_dist.append(avg_ic)

                # Look at the distance of the i_c-th cluster in ONE k means realization to ALL the remaining k-mean realizations
                for other_centers in centers[:][i_c+1::]:
                    for other_center in other_centers:
                        dist = t.distance(this_center, other_center)
                        center_dist.append(dist)
    
                # Do try-except so that it is safe for the last step (comparing the last item with itself)
                try:
                    min_scatter = min(center_dist)
                    minima.append(min_scatter)

                    if verbose:
                        print(f'{i_c}) Center: {this_center}, min scatter: {min_scatter}')
                except:
                    pass
        
        minimum = np.mean(minima)
        stddev = np.std(minima)
        avgdist = np.median(all_dist)
        print(f'k={n_clusters}, average center scatter={minimum}, stddev on center scatter: {stddev}, median interdist: {avgdist}, rescaled scatter: {minimum/avgdist}')

        results[n_clusters-2, 0] = n_clusters
        results[n_clusters-2, 1] = minimum
        results[n_clusters-2, 2] = stddev
        results[n_clusters-2, 3] = avgdist
        results[n_clusters-2, 4] = minimum/avgdist

    print(results)

    if f_out != None:
        for i_col, col in enumerate(columns):
            df_print[col] = results[:, i_col]
        
        print(df_print.head())

        sns.lineplot(x=df_print[columns[0]], y=df_print[columns[1]])
        sns.lineplot(x=df_print[columns[0]], y=df_print[columns[4]])
        plt.legend(labels=['Center Scatter', 'Center Scatter Norm.'])
        plt.show()



def evaluate_metrics(data=None, n_clusters_max=None, n_init=10):

    sil_score = []
    ch_score = []
    ks = []

    n_data = int(len(data) / 1000)
    data = data.sample(n_data, random_state = 1389)

    for n_clusters in range(2, n_clusters_max+1):

        ks.append(n_clusters)
        kmeans = KMeans(n_clusters = n_clusters, n_init = n_init)
        kmeans.fit(data)

        X = data.values
        labels = kmeans.fit_predict(X)

        # This is the average score computed among the individual ones
        s_score = silhouette_score(X, labels)  
        c_score = calinski_harabasz_score(X, labels)
    
        print(f'Silhouette score for n_clusters = {n_clusters} is {s_score}, CH score is {c_score}')

        sil_score.append(s_score)
        ch_score.append(c_score)
        
    plt.cla()
    plt.clf()
    plt.xlabel('k')
    plt.ylabel('CH-score')
    plt.plot(ks, ch_score, color='blue')
    plt.savefig('output/ch_score.png')

    plt.cla()
    plt.clf()
    plt.xlabel('k')
    plt.ylabel('Silhouette-score')
    plt.plot(ks, sil_score, color='blue')
    plt.savefig('output/sil_score.png')
    plt.cla()
    plt.clf()

    print('Elbow Method score...')
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(2, n_clusters_max))

    visualizer.fit(X)        
    visualizer.show()
    plt.show()

    return kmeans


def plot_vweb(data=None, fout=None, thresh=0.0, grid=64, box=100.0, thick=2.0):
    
    z_min = box * 0.5 - thick
    z_max = box * 0.5 + thick

    data = data[data['z'] > z_min]
    data = data[data['z'] < z_max]

    shift = box * 0.5
    data['x'] = data['x'].apply(lambda x: x - shift)
    data['y'] = data['y'].apply(lambda x: x - shift)

    if box > 1e+4:
        data['x'] = data['x'] / 1e+3
        data['y'] = data['y'] / 1e+3
        data['z'] = data['z'] / 1e+3
        shift = shift / 1e+3

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
    elif grid == 256:
        size = 3

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
    plt.title('$\log_{10}\Delta_m', fontsize=fontsize)
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
#web_file = 'vweb_00_10.000032.Vweb-csv'; str_grid = '_grid32'; grid = 32
web_file = 'vweb_00_10.000064.Vweb-csv'; str_grid = '_grid64'; grid = 64
#web_file = 'vweb_00_10.000128.Vweb-csv'; str_grid = '_grid128'; grid = 128

#web_file = 'vweb_128_.000128.Vweb-csv'; str_grid = '_grid128box500'; grid = 128
#web_file = 'vweb_256_.000256.Vweb-csv'; str_grid = '_grid256box500'; grid = 256

#box = 500.0e+3; thick = 7.0e+3
#box = 500.0; thick = 5.0
box = 100.0; thick = 2.0

web_df = pd.read_csv(file_base + web_file)

#plot_new_form(data=web_df, f_out='output/evs_new_form')

n_clusters = 10
n_init = 1

if normalize == True:
    #norm = 1/1024.0 
    norm = 1.0e-3   # kpc to Mpc
    print('Norm: ', norm) 

    web_df['l1'] = web_df['l1'] / norm
    web_df['l2'] = web_df['l2'] / norm
    web_df['l3'] = web_df['l3'] / norm


if plotStd == True: 

    f_out = 'output/kmeans_oldvweb' + str_grid
    #norm = web_df['l1'].max()

    for thresh in [0.0, 0.1]:
        plot_vweb(fout=f_out, data=web_df, thresh=thresh, grid=grid, box=box, thick=thick)

web_df['logdens'] = np.log10(web_df['dens'])
#print(web_df.head())

cols_select = ['l1', 'l2', 'l3']; vers = ''; str_kmeans = r'$k$-means $\lambda$s'
#cols_select = ['l1', 'l2', 'l3', 'dens']; vers = 'd'; str_kmeans = r'$k$-means $\lambda$s, \Delta_m'
#cols_select = ['l1', 'l2', 'l3', 'logdens']; vers = 'ld'; str_kmeans = r'$k$-means $\lambda$s, $\log_{10}\Delta_m$'
#cols_select = ['l1', 'l2', 'l3', 'dens', 'Vx', 'Vy', 'Vz']; vers = 'd_vx'

web_ev_df = web_df[cols_select]

if evalMetrics == True:

    n_ks = 10
    rescale = 100
    f_out = 'output/kmeans_stability' + str_grid

    kmeans = evaluate_metrics(data=web_ev_df, n_clusters_max=n_clusters, n_init=n_init)
    #kmeans_stability(data=web_ev_df, n_clusters_max=n_clusters, n_ks=n_ks, rescale_factor=rescale, f_out=f_out)
    
else:
    kmeans = KMeans(n_clusters = n_clusters, n_init = n_init)
    kmeans.fit(web_ev_df)

centers = kmeans.cluster_centers_
web_df['env'] = kmeans.labels_

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

if plotNew == True:
    for il, cl in enumerate(kmeans.labels_):
        cols.append(colors_sort[cl])

    horizont = [web_ev_df['l3'].min(), web_ev_df['l1'].max()]
    vertical = [web_ev_df['l2'].min(), web_ev_df['l2'].max()]
    diagonal = [web_ev_df['l3'].min(), web_ev_df['l1'].max()]
    zeros = [0.0, 0.0]

    print('Plotting in new format...')
    plt.plot(horizont, zeros, color = 'black')
    plt.plot(zeros, vertical, color = 'black')
    plt.plot(diagonal, diagonal, color = 'blue')

    plt.scatter(web_ev_df['l1'], web_ev_df['l2'], c = cols, marker = 'v') #, size = size1)
    plt.scatter(web_ev_df['l3'], web_ev_df['l2'], c = cols, marker = '+') #, size = size2)
    plt.xlabel(r'$\lambda_1, \lambda_3$')
    plt.ylabel(r'$\lambda_2$')
    plt.tight_layout()
    f_out = out_evs_new + str_grid + '.png'
    plt.savefig(f_out)
    print('Done.')
 
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



