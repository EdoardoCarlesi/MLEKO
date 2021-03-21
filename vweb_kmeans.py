"""
    MLEKO
    Machine Learning Environment for KOsmology 

    (C) Edoardo Carlesi 2020
    https://github.com/EdoardoCarlesi/MLEKO
"""

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import webtools as wt
import pandas as pd
import seaborn as sns
import numpy as np
import tools as t


def order_by_delta(n_clusters=4, web_df=None, centers=None):
    """ Return color codes sorted by median density of environment type """

    if n_clusters == 2:
        colors = ['lightgrey', 'black']
        envirs = ['void', 'knot']
    elif n_clusters == 3:
        colors = ['lightgrey', 'darkgrey', 'red']
        envirs = ['underdense', 'filament', 'knot']
    elif n_clusters == 4:
        number = [0, 1, 2, 3]
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
    number_sort = []

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
        number_sort.append(number[index[0][0]])

        num_str = '& $ %.3f $ & $%.3f$ ' % (deltas[i], n/ntot)
        tab_str = env_str + num_str
        print(tab_str)

    print('\n')

    #for i in range(0, n_clusters):
    #    c_str = ' & $ %.3f $  &  $ %.3f $ & $ %.3f $ ' % (centers[i,0], centers[i, 1], centers[i, 2])
    #    line_str = envirs_sort[i] + c_str
    #    print(line_str)

    return envirs_sort, colors_sort, number_sort


def web_to_csv(web_f=None, web_csv=None):
    """" Convert ASCII Vweb file to CSV format """

    print('Converting to csv: ', web_csv)
    columns = ['dens', 'Vx', 'Vy', 'Vz', 'l1', 'l2', 'l3', 'e1x', 'e1y', 'e1z', 'e2x', 'e2y', 'e2z', 'e3x', 'e3y', 'e3z']
    vweb = pd.read_csv(web_f, sep=' ')
    print(vweb.head())
    vweb.columns = columns
    vweb = gen_coord(data=vweb)
    vweb.to_csv(web_csv)
    print('Done.')

    return vweb 


def gen_coord(data=None, grid=128, box=100.0, cols=['x','y','z']):
    """ Generate the coordinates on the x,y,z grid for the vweb """
 
    print('Generating x,y,z coordinates for the v-web')
    cell = box / grid
    half = cell * 0.5
    n_dim = len(data)
    x_arr = np.zeros(n_dim)
    y_arr = np.zeros(n_dim)
    z_arr = np.zeros(n_dim)

    for i in range(0, grid):
        for j in range(0, grid):
            for k in range(0, grid):
                ind = i + j * grid + k * grid * grid 
                x_arr[ind] = i * cell + half
                y_arr[ind] = j * cell + half
                z_arr[ind] = k * cell + half

    data['x'] = x_arr
    data['y'] = y_arr
    data['z'] = z_arr
    print('Done')

    return data


def plot_cmp():
    """ Compare the results of the k-means vs. standard threshold """

    cmp_file = 'output/vweb_std.dat'
    data = pd.read_csv(cmp_file, dtype=float)

    print(data.head(50))
    print(data.info())

    cols = ['eq1', 'eq2', 'eq3', 'eq4']
    avg = np.zeros(len(data))
    #data['avg'] = data[cols].apply(lambda x: np.sum(x))
    #print(data['avg'])
    for col in cols:
        for i in range(0, len(data)):
            avg[i] += data[col].values[i]

    avg[:] = avg[:] / 4.0

    data['avg'] = avg

    avgMax = data['avg'].max()
    totMax = data['tot'].max()
    lAvgMax = data[data['avg'] == avgMax]['lambda']
    lTotMax = data[data['tot'] == totMax]['lambda']
    print('AvgMax: ', lAvgMax)
    print('TotMax: ', lTotMax)
    
    fontsize = 30
    plt.figure(figsize=(10, 8))
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.grid(False)
    plt.plot(data['lambda'].values, data['tot'].values, color='black', label='tot', linewidth=5)
    plt.plot(data['lambda'].values, data['avg'].values, color='blue', label='avg', linewidth=5)
    plt.legend(fontsize=fontsize)
    plt.xlabel(r'$\lambda_{thr}$', fontsize=fontsize)
    plt.ylabel('Fraction of matching cells', fontsize=fontsize)
    plt.tight_layout()
    #plt.show()
    plt.savefig('output/kmeans_threshold_cmp.png')
    plt.cla()
    plt.clf()
    plt.close()


def loop_resample_web():
    base_path = ''
    base_out = ''
    suffix = ''
    init_web = 0
    end_web = 10

    for i_web in (init_web, end_web):
        this_file = base_path + str(i_web) + suffix
        this_web = pd.read_csv(this_file)
        random_state = 101
        n_sample = int(len(this_web) / 10)
        this_web = this_web.sample(n_sample)
        #wt.evaluate_metrics(data=web_df[['l1', 'l2', 'l3']], elbow=True)


if __name__ == "__main__":
    """
        MAIN PROGRAM - compute K-Means
    """

    # Program Options: what should we run / analyze?
    normalize = False
    evalMetrics = False
    doYehuda = False

    plotNew = False
    plotStd = True
    plot3d = False
    plotKLV = False
    plotEVs = False
    plotLambdas = False

    read_kmeans = True
    do_cmp = False

    #file_base = '/home/edoardo/CLUES/DATA/Vweb/512/CSV/'
    #file_base = '/home/edoardo/CLUES/DATA/Vweb/FullBox/'
    file_ascii = '/home/edoardo/CLUES/TEST_DATA/VWeb/vweb_2048.000128.Vweb-ascii'
    #file_base = '/home/edoardo/CLUES/TEST_DATA/VWeb/'
    #file_base = '/home/edoardo/CLUES/DATA/VWeb/'
    file_base = '/home/edoardo/CLUES/DATA/LGF/512/05_14/'
    #web_file = 'vweb_00_10.000032.Vweb-csv'; str_grid = '_grid32'; grid = 32
    #web_file = 'vweb_00_10.000064.Vweb-csv'; str_grid = '_grid64'; grid = 64
    #web_file = 'vweb_00_10.000128.Vweb-csv'; str_grid = '_grid128'; grid = 128
    #web_file = 'vweb_00.000128.Vweb-csv'; str_grid = '_grid128'; grid = 128
    #web_file = 'vweb_2048.000128.csv'; str_grid = '_grid128'; grid = 128
    web_file = 'vweb_512_128_054.000128.Vweb.csv'; str_grid = '_grid128'; grid = 128
    #web_file = 'vweb_2048.000128.Vweb-ascii'; str_grid = '_grid128'; grid = 128
    #web_file = 'vweb_25_15.000128.Vweb-csv'; str_grid = '_grid128'; grid = 128
    #web_file = 'vweb_00_00.000128.Vweb-csv'; str_grid = '_grid128'; grid = 128; normalize = True

    web_kmeans_file = file_base + 'vweb_kmeans.csv'

    #box = 500.0e+3; thick = 7.0e+3
    #box = 500.0; thick = 5.0
    box = 100.0; thick = 2.0

    wt.elbow_visualize()
    plot_cmp()

    web_df = pd.read_csv(file_base + web_file, dtype=float)
    web_df = gen_coord(data=web_df)

    # Rescale the coordinates
    web_df['x'] = web_df['x'].values - box * 0.5
    web_df['y'] = web_df['y'].values - box * 0.5

    n_clusters = 4
    n_init = 1

    #threshold_list = [0.0, 0.1, 0.2]
    threshold_list = [0.22, 0.26]
    #threshold_list = []

    # Check out that the vweb coordinates should be in Mpc units
    if normalize == True:
        norm = 1.0e-3   # kpc to Mpc
        print('Norm: ', norm) 

        web_df['l1'] = web_df['l1'] / norm
        web_df['l2'] = web_df['l2'] / norm
        web_df['l3'] = web_df['l3'] / norm

    if plotStd == True: 

        for thresh in threshold_list:
            f_out = 'output/kmeans_oldvweb' + str_grid + '_l' + str(thresh)
            title_str = r'$\lambda _{thr}=$' + str(thresh)
            web_df = wt.plot_vweb(fout=f_out, data=web_df, thresh=thresh, grid=grid, box=box, thick=thick, do_plot=True, title=title_str)
            wt.plot_lambda_distribution(data=web_df, base_out=f_out, x_axis=True)

    cols_select = ['l1', 'l2', 'l3']; vers = ''; str_kmeans = r'$k$-means $\lambda$s'

    n_clusters = 4
    n_init = 101

    if read_kmeans:

        print('Loading k-means from output/kmeans.pkl')
        web_df = pd.read_csv(web_kmeans_file)
        kmeans = pickle.load(open('output/kmeans.pkl', 'rb'))
        centers = kmeans.cluster_centers_

    else:

        print('Running kmeans...')
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
        kmeans.fit(web_df[cols_select])
        centers = kmeans.cluster_centers_
        pickle.dump(kmeans, open('output/kmeans.pkl', 'wb'))
        web_df['env'] = kmeans.labels_
        web_df.to_csv(web_kmeans_file)
        print('Done.')
        
    #wt.evaluate_metrics(data=web_df[['l1', 'l2', 'l3']], elbow=True)

    f_out = 'output/kmeans_' + str_grid 
    envirs_sort, colors_sort, number_sort = order_by_delta(n_clusters=4, web_df=web_df, centers=centers)
    wt.plot_lambda_distribution(data=web_df, base_out=f_out, x_axis=True)
    title_str = r'$k$-means'
    wt.plot_vweb(data=web_df, fout='output/vweb_kmeans_128.png', grid=128, use_thresh=False, ordered_envs=number_sort, title=title_str)

    web_df['envk_std'] = kmeans.labels_
    web_df = wt.order_kmeans(data=web_df)

    if plot3d:
        out_evs_3d = 'output/kmeans_3d_' + vers
        f_out = out_evs_3d + str_grid + '.png'
        wt.plot3d(labels=kmeans.labels_, data=web_df, f_out=f_out)

    if do_cmp:

        for col in cols_select:
            med = np.median(web_df[col])
            std = np.std(web_df[col])
            str_print = '%.3f \pm %.3f & ' % (med, std)

        thresh = [i * 0.02 for i in range(0, 30)]

        for th in thresh:
            web_df = wt.std_vweb(data=web_df, thresh=th)
            wt.compare_vweb_kmeans(vweb=web_df, l=th)


'''
    if doYehuda:

        n_clusters_tot = [2, 3, 4, 5, 6, 7, 8, 9, 10] 
        random_state = 1

        for n_clusters in n_clusters_tot:
            kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
            kmeans.fit(data)
            web_df['class'] = kmeans.labels_

            fout = 'output/vweb_' + str(grid) + '_k' + str(n_clusters) + '_rand_state' + str(random_state) + '.txt'
            print(n_clusters, fout)
            web_df.to_csv(path_or_buf=fout, sep='\t', index=False, float_format='%.3f')

    if evalMetrics == True:

        n_ks = 10
        rescale = 1
        f_out = 'output/kmeans_stability' + str_grid
        #n_clusters_tot = [2, 3] 
        n_clusters_tot = [2, 3, 4, 5, 6, 7, 8, 9, 10] 
        #kmeans = evaluate_metrics(data=web_ev_df, n_clusters_max=n_clusters, n_init=n_init)
        #kmeans_stability(data=web_ev_df, n_clusters_max=n_clusters, n_ks=n_ks, rescale_factor=rescale, f_out=f_out)
    
        prior='lognorm'
        #prior='gauss'
        #prior='flat'
        mode='ordered'
        #mode='simple'
        #data_rand = wt.generate_random(data=data, grid=grid, prior=prior, verbose=True, mode=mode)
        #data_rand = wt.generate_random(data=data, grid=grid, prior='gauss', verbose=True, mode='ordered')
        #print(data_rand.head())
        #data_rand.to_csv('output/random_web_128.txt', index=False)
        #data_rand = pd.read_csv('output/random_web_128.txt')
        #print(data_rand.head())

        k_all = []
        k_std = []
        k_rand = []
        #wt.evaluate_metrics(data=data_rand, n_clusters_max=10, n_init=0, rescale_factor=1, visualize=True, elbow=True)
        wt.evaluate_metrics(data=data, n_clusters_max=10, n_init=0, rescale_factor=1, visualize=True, elbow=True)
#def evaluate_metrics(data=None, n_clusters_max=None, n_init=10, rescale_factor=10, visualize=False, elbow=True):

        n_clusters_tot = 0
        for n_clusters in n_clusters_tot:
            kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
            kmeans.fit(data)
                
            kmeans_rand = KMeans(n_clusters=n_clusters, n_init=n_init)
            kmeans_rand.fit(data_rand)

            centers_rand = kmeans_rand.cluster_centers_
            centers = kmeans.cluster_centers_

            data_rand['label'] = kmeans_rand.labels_
            data['label'] = kmeans.labels_

            wsc = 0.0
            wsc_rand = 0.0 
        
            for i in range(0, n_clusters):

                n = len(data[data['label'] == i])
                c = centers[i]
                c = np.reshape(c, (3, 1))
                vals = data[data['label'] == i][cols_select].T.values - c
                wsc += np.sum(np.sqrt((vals)**2.0))

                n_rand = len(data_rand[data_rand['label'] == i])
                c_rand = centers_rand[i]
                c_rand = np.reshape(c_rand, (3, 1))
                vals_rand = data_rand[data_rand['label'] == i][cols_select].T.values - c_rand
                wsc_rand += np.sum(np.sqrt((vals_rand)**2.0))

            print(f'ROUND k={i+1}/{n_clusters}')
            print(n_clusters, wsc/n_clusters, wsc_rand/n_clusters)

            k_all.append(n_clusters)
            k_std.append(wsc/n_clusters)
            k_rand.append(wsc_rand/n_clusters)

                #diffs_rand = data_rand[data_rand['label'] == i].values
                #print(i, 'Std: ', len(diffs))
                #print(i, 'Ran: ', len(diffs_rand))
                #print(f'Cluster i has stddev: {stddev}')

            data_rand.drop(labels='label', inplace=True, axis=1)
            data.drop(labels='label', inplace=True, axis=1)

        k_all = np.array(k_all)
        k_std = np.array(k_std)
        k_rand = np.array(k_rand)

        print(k_all)
        print(k_std)
        print(k_rand)

        #plt.plot(k_all, k_std)
        #plt.plot(k_all, k_rand)
        figname = 'output/gap_stats_' + prior + '_' + mode + '_' + str(grid) + '.png'
        plt.xlabel('k')
        plt.ylabel('WSCdata - WSCrand')
        plt.plot(k_all, np.abs(k_rand-k_std))
        plt.title('Gap Statistics (' + prior + ', ' + mode + ')')
        plt.tight_layout()
        plt.savefig(figname)
        plt.show()

        exit()


    else:
        kmeans = KMeans(n_clusters = n_clusters, n_init = n_init)
        kmeans.fit(web_ev_df)

kmeans = KMeans(n_clusters = n_clusters, n_init = n_init)
kmeans.fit(web_ev_df)
centers = kmeans.cluster_centers_
web_df['env'] = kmeans.labels_

vers = vers + '_k' + str(n_clusters)
out_evs_dist = 'output/kmeans_vweb_' + vers
out_dens_dist = 'output/kmeans_dens_' + vers
out_web_slice = 'output/kmeans_web_lv_' + vers

cols = []

envirs_sort, colors_sort = order_by_delta(n_clusters=4, web_df=web_df)
   
out_evs_new = 'output/kmeans_new_' + vers
f_out = out_evs_new + str_grid + '.png'
wt.plot_new(labels=kmeans.labels_, data=web_ev_df, f_out=f_out)
# TODO there should be some loop on the environments here
env_type = ''
f_out_base = ''
wt.plot_eigenvalues_per_environment_type(data=None, env_type=None, out_base=None, grid=None) 


#if plotLambdas == True:
#def plot_lambda_distribution(data=None, grid=None, base_out=None, envirs=None):
#def plot_local_volume_density_slice(data=None, box=100, file_out=None, title=None):

'''
