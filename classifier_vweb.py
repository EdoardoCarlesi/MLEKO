'''
    CluesML
    Machine Learning tools for the CLUES project

    (C) Edoardo Carlesi 2020
    https://github.com/EdoardoCarlesi/CluesML
'''

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tools as t

file_base = '/home/edoardo/CLUES/DATA/Vweb/512/CSV/'
web_file = 'vweb_00_10.000032.Vweb-csv'; str_grid = '_grid32'
#web_file = 'vweb_00_10.000064.Vweb-csv'; str_grid = '_grid64'
#web_file = 'vweb_25_15.000064.Vweb-csv'
#web_file = 'vweb_01_10.000064.Vweb-csv'
#web_file = 'vweb_00_10.000128.Vweb-csv'

web_df = pd.read_csv(file_base + web_file)
print(web_df.head())

cols_select = ['l1', 'l2', 'l3']

web_ev_df = web_df[cols_select]

kmeans = KMeans(n_clusters = 4, n_init = 10)
kmeans.fit(web_ev_df)

#print(kmeans.cluster_centers_)
centers = kmeans.cluster_centers_

#print(type(kmeans.labels_))

web_df['env'] = kmeans.labels_
ntot = len(web_df)

colors = ['white', 'grey', 'black', 'red']
envirs = ['void', 'sheet', 'filament', 'knot']

deltas = np.zeros((4))

for i in range(0, 4):
    deltas[i] = np.median(web_df[web_df['env'] == i]['dens'].values)
    print(i, deltas[i])

deltas_sort = np.sort(deltas)

for i in range(0, 4):
    n = len(web_df[web_df['env'] == i]) 
    #print(n, ' perc: ', n/ntot)

    env_str = str(i) 
    evs_df = web_df[web_df['env'] == i]

    center_str = '%.3f %.3f %.3f' % (centers[i, 0], centers[i, 1], centers[i, 2])

    plt.xlabel('eigenvalues')
    plt.title('Environment type ' + env_str + ', center: ' + center_str)

    for col in cols_select:
        sns.distplot(evs_df[col])

        f_out = 'output/kmeans_vweb_' + env_str + str_grid + '.png'
        plt.savefig(f_out)

    plt.clf()
    plt.cla()

    plt.xlabel(r'$\delta$')
    deltam = '%.3f' % deltas[i]
    plt.title('Environment type ' + env_str + ', median delta: ' + deltam)
    sns.distplot(evs_df['dens'])
    f_out = 'output/kmeans_dens_' + env_str + str_grid + '.png'
    plt.savefig(f_out)
    plt.clf()
    plt.cla()
    #plt.show(block = False)
    #plt.pause(2)
    #plt.close()

# Plot a slice of the local volume


z_min = 48.00
z_max = 52.00

web_df = web_df[web_df['z'] > z_min]
web_df = web_df[web_df['z'] < z_max]

plt.xlabel(r'SGX \quad $[h^{-1} Mpc]$')
plt.ylabel(r'SGY \quad $[h^{-1} Mpc]$')
plt.xlim([40, 60])
plt.ylim([40, 60])
#plt.plot(web_df['x'], web_df['y'], c = colors[web_df['env'].values])
#f_out = 'output/kmeans_web_lv' + str_grid + '.png'
#plt.savefig()


'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(web_ev_df['l1'], web_ev_df['l2'], web_ev_df['l3'], c = kmeans.labels_)
plt.show()
'''
