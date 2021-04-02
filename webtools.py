"""
    MLEKO
    Machine Learning Environment for KOsmology

    (C) Edoardo Carlesi 2020
    https://github.com/EdoardoCarlesi/MLEKO
"""

import matplotlib.pyplot as plt
import matplotlib.axis as ax
import matplotlib 
import pandas as pd
import seaborn as sns
import numpy as np
import tools as t
import pickle as pkl
import read_files as rf

from sklearn.metrics import silhouette_score, homogeneity_score, calinski_harabasz_score, silhouette_samples
from yellowbrick.cluster import KElbowVisualizer
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def find_env(l1, l2, l3, thresh):
    """ Find environment type """

    if l1 < thresh:
        return 0.
    elif l1 > thresh and l2 < thresh:
        return 1.
    elif l2 > thresh and l3 < thresh:
        return 2.
    elif l3 > thresh:
        return 3.

    
def plot_eigenvalues_per_environment_type(data=None, env_type=None, out_base=None, grid=None): 
    """ Plot the three eigenvalues distributions for a given environment type """

    evs = data[data['env'] == env_type]

    # Only plot the first three axes i.e. the eigenvalues
    for col in cols_select[0:3]:
        sns.distplot(evs_df[col])
    
    file_out = out_base + '_evs.png'

    print(f'Plotting eigenvalues per environment type to: {file_out}')
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
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    deltam = '%.3f' % deltas[i]
    plt.title(env_str + r': $ \bar \Delta_m = $' + deltam + ' $, k=$' + str(n_clusters))
    sns.distplot(np.log10(evs['dens']), bins=nbins)

    plt.xlabel(r'$\log_{10}\Delta _m$', fontsize=fontsize)
    file_out = out_base + '_dens.png'
    print(f'Plotting density per environment type to: {file_out}')
    plt.savefig(file_out)
    plt.clf()
    plt.cla()


def assign_halos_to_environment_type(snap_ini=0, snap_end=5, vweb=None, kmeans=None):
    """ Given a snapshot / series of snapshots read the halo catalog and make it into different environment types """

    base_path = '/home/edoardo/CLUES/DATA/FullBox/17_11/snapshot_127.'
    suffix = '.z0.000.AHF_halos'
    
    # We will save the halos into a list of the form [halo_mass, type_kmeans, type_vweb]
    halo_env = []

    # Vweb and kmeans are arrays with index = ix + n * iy + n * n * iz
    x_col = ['Xc(6)', 'Yc(7)', 'Zc(8)']
    m_col = 'Mvir(4)'
    fac = 1.0e+3

    for snap_i in range(snap_ini, snap_end):
        snap_str = '%04d' % snap_i
        this_ahf = base_path + snap_str + suffix

        if snap_i == 0:
            this_halos = rf.read_ahf_halo(this_ahf)
            columns = list(this_halos.columns)

            # The columns are not sorted correctly so we need to add a dummy label at the beginning and remove the last one for consistency
            columns.insert(0, 'Dummy')
            columns.remove(columns[-1])

        else:
            this_halos = rf.read_ahf_halo(this_ahf, use_header=True, header=columns)
        
        for i, row in this_halos.iterrows():
            this_m = row[m_col]
            this_x = row[x_col].T / fac
            index = find_nearest_node_index(x=this_x, grid=128, box=100.0)

            vweb_type = vweb[index]
            kmeans_type = kmeans[index]

            this_halo_env = [this_m, kmeans_type, vweb_type]
    
            halo_env.append(this_halo_env)

            print(index, this_x.values, this_m/1.e+10, vweb_type, kmeans_type)
            #print(this_halos.head())

    cols = np.array(halo_env)
    data = pd.DataFrame(data=cols, columns=['M', 'kmeans', 'vweb'])
    
    print(data.head())

    return data


def find_nearest_node_index(x=None, grid=None, box=None):
    """ Given a point x in space, find the nearest grid point once a grid has been placed on the box """

    cell = box / grid
    ix = np.floor(x[0] / cell)
    iy = np.floor(x[1] / cell)
    iz = np.floor(x[2] / cell)

    index = int(ix + grid * iy + grid * grid * iz)

    return index


def plot_new_format(data=None, f_out=None, labels=None):
    """ Plot the point distribution according to Yehuda's new suggestion """

    for il, cl in enumerate(labels):
        cols.append(colors_sort[cl])

    horizont = [data['l3'].min(), data['l1'].max()]
    vertical = [data['l2'].min(), data['l2'].max()]
    diagonal = [data['l3'].min(), data['l1'].max()]
    zeros = [0.0, 0.0]

    print('Plotting in new format...')
    plt.plot(horizont, zeros, color = 'black')
    plt.plot(zeros, vertical, color = 'black')
    plt.plot(diagonal, diagonal, color = 'blue')

    plt.scatter(web_ev_df['l1'], web_ev_df['l2'], c = cols, marker = 'v') 
    plt.scatter(web_ev_df['l3'], web_ev_df['l2'], c = cols, marker = '+') 
    plt.xlabel(r'$\lambda_1, \lambda_3$')
    plt.ylabel(r'$\lambda_2$')
    plt.tight_layout()
    plt.savefig(f_out)
    print('Done.')
 

def plot3d(labels=None, data=None, f_out=None):
    """
    Plot a 3d distribution of points
    - labels can be kmeans.labels_ , it is an integer describing the class each point belongs to
    - data is the v-web eigenvalues dataframe
    """

    #colors = ['g', 'g', 'b', 'r']
    #colors = ['lightgrey', 'grey', 'black', 'red']
    #labels = ['void', 'sheet', 'filament', 'knot']
    colors = ['lightgrey', 'black', 'grey', 'red']
    labels = ['void', 'filament', 'sheet', 'knot']
    order = [0, 2, 1, 3]
    c_rgb = []

    for color in colors:
        c_rgb.append(matplotlib.colors.to_rgb(color))

    print('Plotting in 3D particle distribution...')
    fig = plt.figure(figsize=(6,5))
    fontsize = 10
    ax = fig.add_subplot(111, projection='3d')
    #plt.grid(False)

    for i, color in zip(order, colors):
        datac = data[data['env'] == float(i)]
        ax.scatter(datac['l1'], datac['l2'], datac['l3'], color=c_rgb[i], label=labels[i])
        #ax.scatter(data['l1'], data['l2'], data['l3'], c = labels, cmap=cm)

    #ax.scatter(data['l1'], data['l2'], data['l3'], c = labels, cmap=cm)

    #plt.xticks(fontsize=fontsize)
    #plt.yticks(fontsize=fontsize)
    #plt.zticks(fontsize=fontsize)
    ax.set_xlabel(r'$\lambda_1$', fontsize=fontsize)
    ax.set_ylabel(r'$\lambda_2$', fontsize=fontsize)
    ax.set_zlabel(r'$\lambda_3$', fontsize=fontsize)
 
    plt.legend(fontsize=fontsize, frameon=True, framealpha=1.0)
    plt.tight_layout()
    plt.savefig(f_out)
    plt.clf()
    plt.cla()
    print('Done.')


def wss(data=None, centers=None, labels=None):
    """ Within sum of squares """
    # TODO   
    pass


@t.time_total
def generate_random(data=None, grid=None, prior='flat', verbose=False, mode='simple'):
    """ Generate a random distribution of eigenvalues """

    print(f'Generating a random distribution of {grid}^3 points using a {prior} prior...')
    n_pts = grid * grid * grid

    means = []
    stddevs = []
    new_col = []
    lows = []
    highs = []

    for col in data.columns:
        means.append(data[col].mean())
        stddevs.append(data[col].std())
        new_col.append(col)
        low = data[col].mean() - 2 * data[col].std()
        high = data[col].mean() + 2 * data[col].std()
        lows.append(low)
        highs.append(high)


    l2s = []
    l3s = []

    if prior == 'flat':
        l1s = np.random.uniform(size=n_pts, low=lows[0], high=highs[0])
        l2s = np.random.uniform(size=n_pts, low=lows[1], high=highs[1])
        l3s = np.random.uniform(size=n_pts, low=lows[2], high=highs[2])
        
    elif prior == 'gauss':
        l1s = np.random.normal(size=n_pts, loc=means[0], scale=stddevs[0])
        l2s = np.random.normal(size=n_pts, loc=means[1], scale=stddevs[1])
        l3s = np.random.normal(size=n_pts, loc=means[2], scale=stddevs[2])

    elif prior == 'lognorm':
        l1s = np.random.lognormal(size=n_pts, mean=means[0], sigma=stddevs[0])
        l2s = np.random.lognormal(size=n_pts, mean=means[1], sigma=stddevs[1])
        l3s = np.random.lognormal(size=n_pts, mean=means[2], sigma=stddevs[2])
    else:
        print(f'Prior type: {prior} is not implemented. Exit program.')
        exit()

    if mode == 'ordered':
        for i in range(0, len(l1s)):
            l1 = l1s[i]
            l2 = l2s[i]
            l3 = l3s[i]

            if l3 > l2:
                l3s[i] = l2
                l2s[i] = l3

            if l2s[i] > l1:
                l1s[i] = l2s[i]
                l2s[i] = l1
    else:
        pass
 
    # Create a dataframe with the newly generated random values of the eigenvalues
    #global rand_data
    rand_data = pd.DataFrame()
    rand_data[new_col[0]] = np.array(l1s)
    rand_data[new_col[1]] = np.array(l2s)
    rand_data[new_col[2]] = np.array(l3s)

    if verbose:
        check_distribution(l1s)
        check_distribution(l2s)
        check_distribution(l3s)
        print(rand_data.head())

    return rand_data


@t.time_total
def evaluate_metrics(data=None, n_clusters_max=10, n_init=10, rescale_factor=10, visualize=False, elbow=True):
    """ Evaluate different metrics to estimate the best k for the cluster number """

    print('evaluate_metrics()')

    if elbow == False:
        sil_score = []
        ch_score = []
        ent_score = []
        ks = []

        n_data = int(len(data) / rescale_factor)
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
            e_score = entropy(labels = labels, k = n_clusters)
            print(data.head())
        
            print(f'Silhouette score for n_clusters = {n_clusters} is {s_score}, CH score is {c_score}, Entropy is {e_score}')

            sil_score.append(s_score)
            ch_score.append(c_score)
            ent_score.append(e_score)
        
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

        plt.rcParams.update({"text.usetex": True})
        plt.cla()
        plt.clf()
        plt.xlabel('k')
        plt.ylabel('Entropy')
        plt.title(r'$H(k) = - \sum_{i=0}^{k}(n_i / n_{tot}) log(n_i / n_{tot})$')
        plt.plot(ks, ent_score, color='blue')
        plt.savefig('output/entropy_score.png')
        plt.cla()
        plt.clf()

    if elbow == True:
        print('Elbow Method score...')
        model = KMeans()
        X = data.values
        visualizer = KElbowVisualizer(model, k=(2, n_clusters_max+1), timings=False)
        visualizer.fit(X)        
        print(visualizer.k_scores_)

        if visualize:
            visualizer.show()
            plt.grid(False)
            plt.legend(False)
            plt.show()

        print('Elbow Method score for Calinski-Harabasz...')
        #visualizer = KElbowVisualizer(model, k=(2, n_clusters_max), timings=False, metric='calinski_harabasz', locate_elbow=True)
        #print(visualizer.k_scores_)

        if visualize:
            visualizer.fit(X)        
            visualizer.show()
            plt.show()

    return visualizer.k_scores_


def elbow_diff(scores):
    """ Compute the optimal k peak value """
        
    n_s = len(scores)
    diff = np.zeros(n_s)

    for s in range(1, n_s-1):
        diff[s] = (scores[s-1] - scores[s]) / (scores[s] - scores[s+1]) 
    
    return diff


def elbow_visualize():
    """ Plot the curve to find the optimal k value """

    f_elbow = 'output/elbow.csv'
    
    data = pd.read_csv(f_elbow)
    fac = 1.e+4
    ymin = data['scoreE'].min()
    ymax = data['scoreE'].max()
    xs = [4, 4, 4]
    ys = [1, 100, ymax * 2]

    scores = data['scoreE'].values
    diff = elbow_diff(scores)
    #print(diff * fac)
    #diff = np.array([0., 15714.28571429, 23333.33333333, 12500., 12000., 11000., 8571.42857143, 7333.33333333, 0.])
    #print(diff)

    size=20
    plt.figure(figsize=(9,7))
    plt.grid(False)
    plt.ylim([0.8 * ymin / fac, 1.1 * ymax/fac])
    plt.xlabel('k', fontsize=size)
    plt.ylabel(r'distortion score $\quad$ [$10^4$]', fontsize=size)

    plt.plot(data['k'].values, data['scoreE'].values/fac, color='blue', linewidth=5, label=r'$W(k)$')
    plt.plot(data['k'].values, 10. * diff/fac, color='black', linewidth=3, label=r'$10 \times \Delta W $')
    plt.plot(np.array(xs), np.array(ys)/fac, color='black', linestyle='--')
    plt.xticks(fontsize=size)
    plt.yticks(fontsize=size)
    plt.legend(fontsize=size)
    plt.tight_layout()
    plt.savefig('output/elbow_score.png')
    plt.cla()
    plt.clf()
    plt.close()


def check_env(l1, l2, l3, thresh=None):
    """ Return the env type given three eigenvalues and a threshold value """

    if l1 < thresh:
        return 0
    if l1 > thresh and l2 < thresh:
        return 1
    if l2 > thresh and l3 < thresh:
        return 2
    if l3 > thresh:
        return 3


def std_vweb(data=None, thresh=None): 
    """ Volume filling fractions for the different kinds of nodes   """

    cols = ['l1', 'l2', 'l3']

    data['env'] = data[cols].apply(lambda x: check_env(*x, thresh=thresh), axis=1)

    ntot = len(data)
    env_names = ['void', 'sheet', 'filament', 'knot']
    
    for env in [0, 1, 2, 3]:
        nv = len(data['env'] == env)
        d_med = np.median(data[data['env'] == env]['dens'])
        f_vol = len(data[data['env'] == env]) / nv
         
        str_part = env_names[env] + ' & '
        for col in cols:
            v_std = np.std(data[data['env'] == env][col])
            v_med = np.median(data[data['env'] == env][col])
            
            str_part += '$%.3f \pm %.3f $ & ' % (v_med, v_std)

        str_df = ' $%.3f$ & $%.3f$ \\\ ' % (d_med, f_vol) 
        str_env = str_part + str_df
        print(str_env)

    #print('\hline')

    return data


def plot_densities(data=None, cols=None):
    """ Compare the density distributions in the three schemes """
    
    envs = ['voids', 'sheets', 'filaments', 'knots']
    colors = ['blue', 'green', 'black']
    linestyles = ['-', '--', '-.']
    legends = [r'$\lambda = 0.18$', r'$\lambda_t$ = 0.21', r'$k$-web']
    fontsize=30
    nbins=75
    lwsize=2

    data = data[data['dens'] < 50]
    data['dens'] = np.log10(data['dens'].values)

    for ie, env in enumerate(envs):
        title = env
        plt.figure(figsize=(10, 10))
        plt.grid(False)
        plt.title(title, fontsize=fontsize)
        plt.xlabel(r'$\log _{10} \Delta _M$', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        for ic, col in enumerate(cols):
            delta = data[data[col] == float(ie)]['dens'].values 

            if ie == 0:
                plt.hist(delta, color=colors[ic], alpha=0.3, bins=nbins, density=True, label=legends[ic])
            else:
                plt.hist(delta, color=colors[ic], alpha=0.3, bins=nbins, density=True, label=legends[ic])

            plt.hist(delta, color=colors[ic], histtype='step', lw=lwsize, density=True, bins=nbins) #, linestyle=linestyles[ic], label=legends[ic])
            plt.grid(False)

        fout = 'output/web_dens_' + env + '_cmp.png'
        print('Plotting to:', fout)

        if ie == 0:
            plt.legend(fontsize=fontsize)

        plt.grid(False)
        plt.tight_layout()
        plt.savefig(fout)
        plt.close()
        plt.clf()
        plt.cla()


def compare_vweb_kmeans(vweb=None, l=0.0):
    """ Compare the fraction of volume occupied by different environmental types in the usual vweb classification and in kmeans """

    shared = []
    n_env_shared = np.zeros(4)
    tot_env = np.zeros(4)

    diffs = vweb['env'].values - vweb['envk'].values
    diffs_inds = np.where(diffs == 0)

    n_all = len(vweb)
    #print(f'Global shared values: {l} {len(diffs[diffs_inds])/n_all}')
    
    for i, env in enumerate([0, 1, 2, 3]):
        n_tot = len(vweb[vweb['envk'] == env])
        tmp = vweb[vweb['env'] == env]
        n_shared = len(tmp[tmp['envk'] == env])
        
        n_env_shared[i] = n_shared
        tot_env[i] = n_tot

        #print(f'Env: {env}, Tot: {n_tot}, Shared: {n_shared}, Perc: {n_env_shared/tot_env}')

    average = np.mean(n_env_shared/tot_env)
    total = len(diffs[diffs_inds])/n_all
    print(f'Global shared values: {l},{total},{average},{n_env_shared/tot_env}')

    for env in [0, 1, 2, 3]:
        n_tot = len(vweb[vweb['env'] == env])
        tmp = vweb[vweb['envk'] == env]
        n_shared = len(tmp[tmp['env'] == env])
        
        #print(f'(inverse check) Env: {env}, Tot: {n_tot}, Shared: {n_shared} Perc: {n_shared/n_tot}')

    return average, total


def order_kmeans(data=None, nk=4):    
    """ Rename the eigenvalue labels by increasing matter density """
    
    ds = []

    for env in range(0, nk):
        d_med = np.median(data[data['envk_std'] == env]['dens'].values)
        ds.append(d_med)
 
    ds_sort = np.sort(ds)
    id_order = []

    for i in range(0, nk):
        index = np.where(ds_sort == ds[i])
        id_order.append(index[0][0])
    
    data['envk'] = data['envk_std'].apply(lambda x: id_order[x])

    return data


@t.time_total
def plot_vweb_smooth(data=None, fout=None, thresh=0.0, grid=64, box=100.0, thick=2.0, use_thresh=True, ordered_envs=None, plot_dens=False, do_plot=True, title=None):
    """ Plot the usual vweb using an input threshold and a given dataset """
    
    envs = [0, 1, 2, 3]
    envs = np.array(envs)
    ordered_envs = np.array(ordered_envs)
    #dummy = np.zeros(len(data))
    #data['env'] = dummy

    z_min = box * 0.5 - thick
    z_max = box * 0.5 + thick

    #data_slice = data[data['z'] > z_min]
    #data_slice = data[data['z'] < z_max]
    
    if grid == 128:
        z0 = 50.391
        cell = 0.391
    elif grid == 32:
        z0 = 51.56250
        cell = 1.56250

    data_slice = data[data['z'] == z0]

    print(len(data_slice), grid * grid)

    '''
    '''
    shift = box * 0.5
    #data['x'] = data['x'].apply(lambda x: x - shift)
    #data['y'] = data['y'].apply(lambda x: x - shift)

    if box > 1e+4:
        data_slice['x'] = data_slice['x'] / 1e+3
        data_slice['y'] = data_slice['y'] / 1e+3
        data_slice['z'] = data_slice['z'] / 1e+3
        shift = shift / 1e+3

    if use_thresh:
        print(f'Plotting web with lambda threshold {thresh}')
        voids = data_slice[data_slice['l1'] < thresh]
        sheet = data_slice[(data_slice['l2'] < thresh) & (data_slice['l1'] > thresh)]
        filam = data_slice[(data_slice['l2'] > thresh) & (data_slice['l3'] < thresh)]
        knots = data_slice[data_slice['l3'] > thresh]
        #data_slice['env'] = data_slice[['l1', 'l2', 'l3']].apply(lambda x: find_env(*x, thresh), axis=1)

    else:
        print(f'Plotting web with pre-computed environment class')
        ind_voids = np.where(ordered_envs == 0)
        ind_sheet = np.where(ordered_envs == 1)
        ind_filam = np.where(ordered_envs == 2)
        ind_knots = np.where(ordered_envs == 3)

        voids = data_slice[data_slice['env'] == ind_voids[0][0]]
        sheet = data_slice[data_slice['env'] == ind_sheet[0][0]]
        filam = data_slice[data_slice['env'] == ind_filam[0][0]]
        knots = data_slice[data_slice['env'] == ind_knots[0][0]]


    print(len(voids) + len(sheet) + len(filam) + len(knots))

    n_pts = float(len(data))
    #str_env_types = '%.2f & %.2f & %.2f & %.2f \\\ ' % (len(voids)/n_pts, len(sheet)/n_pts, len(filam)/n_pts, len(knots)/n_pts)
    #print(str_env_types)
    str_dens_types = '%.2f & %.2f & %.2f & %.2f \\\ ' % (voids['dens'].median(), sheet['dens'].median(), filam['dens'].median(), knots['dens'].median())
    print(str_dens_types)
    
    dens_grid = np.zeros((grid, grid))
    
    def x2i(x, y):
        i = int(((x - cell) / box) * grid)
        j = int(((y - cell) / box) * grid)
        return (i, j)

    indexes = dict()

    for i in range(0, grid):
        for j in range(0, grid):
            for k in range(0, grid):
                #index = i + j * grid + k * grid * grid
                index = k + j * grid + i * grid * grid
                indexes[str(index)] = [k, j, i]

    def ind2i(ind):
        inds = indexes[str(ind)]
        return (inds[1], inds[0])

    check_duplicates = []

    for ind, row in voids.iterrows():
        x, y = row[['x', 'y']]
        #i, j = x2i(x, y)
        #print(i, j, x, y)
        i, j = ind2i(ind)
        check_duplicates.append([i, j])
        #print(i, j, ind)
        dens_grid[i, j] = 0
 
    for ind, row in sheet.iterrows():
        x, y = row[['x', 'y']]
        #i, j = x2i(x, y)
        i, j = ind2i(ind)
        check_duplicates.append([i, j])
        dens_grid[i, j] = 2
 
    for ind, row in filam.iterrows():
        x, y = row[['x', 'y']]
        #i, j = x2i(x, y)
        i, j = ind2i(ind)
        check_duplicates.append([i, j])
        dens_grid[i, j] = 3
 
    for ind, row in knots.iterrows():
        x, y = row[['x', 'y']]
        #i, j = x2i(x, y)
        i, j = ind2i(ind)
        check_duplicates.append([i, j])
        dens_grid[i, j] = 4


    non_duplicates = []

    for elem in check_duplicates:
        if elem in non_duplicates:
            pass
        else:
            non_duplicates.append(elem)

    print(len(check_duplicates))
    print(len(non_duplicates))

    #pkl.dump(dens_grid, open('output/dens_grid.pkl', 'wb'))
    #dens_grid = pkl.load(open('output/dens_grid.pkl', 'rb'))
    dens_grid[:, :] = dens_grid[::-1, :]

    print(dens_grid)

    if do_plot:

        fontsize = 25
    
        # Plot the eigenvaule threshold based V-Web
        plt.figure(figsize=(10, 10))
        plt.rcParams["axes.edgecolor"] = "0.0"
        plt.rcParams["axes.linewidth"]  = 1

        plt.xlabel(r'SGX $\quad [h^{-1} Mpc]$', fontsize=fontsize)
        plt.ylabel(r'SGY $\quad [h^{-1} Mpc]$', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        #if use_thresh:
        if title != None:
            plt.title(title, fontsize=fontsize)
        colormap = 'viridis'
        size = 20
        interp='gaussian'
        #interp='spline36'
        #interp='lanczos'
        #interp='none'
        plt.grid(False)
        plt.imshow(dens_grid, cmap=colormap, interpolation=interp, extent=[-50, 50, -50, 50])
        plt.grid(False)
        #plt.pcolormesh(dens_grid) #, cmap=colormap, interpolation=interp)
#        plt.scatter(voids['x'], voids['y'], c='lightgrey', s=size, marker='s')
#        plt.scatter(sheet['x'], sheet['y'], c='grey', s=size, marker='s')
#        plt.scatter(filam['x'], filam['y'], c='black', s=size, marker='s')
#        plt.scatter(knots['x'], knots['y'], c='red', s=size, marker='s')
        plt.rcParams["axes.edgecolor"] = "0.0"
        plt.rcParams["axes.linewidth"]  = 1

        # Save file
        f_out = fout + '_' + str(thresh).replace('.','') + '.png'
        plt.tight_layout()
        print('Saving fig to ', f_out)
        plt.savefig(f_out)
        plt.cla()
        plt.clf()

    return data_slice


@t.time_total
def plot_vweb(data=None, fout=None, thresh=0.0, grid=64, box=100.0, thick=2.0, use_thresh=True, ordered_envs=None, plot_dens=False, do_plot=True, title=None):
    """ Plot the usual vweb using an input threshold and a given dataset """
    
    envs = [0, 1, 2, 3]
    envs = np.array(envs)
    ordered_envs = np.array(ordered_envs)
    #dummy = np.zeros(len(data))
    #data['env'] = dummy

    z_min = box * 0.5 - thick
    z_max = box * 0.5 + thick
    print(z_min, z_max)
    
    data_slice = data[data['z'] > z_min]
    data_slice = data_slice[data_slice['z'] < z_max]

    print(len(data_slice))

    shift = box * 0.5
    #data['x'] = data['x'].apply(lambda x: x - shift)
    #data['y'] = data['y'].apply(lambda x: x - shift)

    if box > 1e+4:
        data_slice['x'] = data_slice['x'] / 1e+3
        data_slice['y'] = data_slice['y'] / 1e+3
        data_slice['z'] = data_slice['z'] / 1e+3
        shift = shift / 1e+3

    if use_thresh:
        print(f'Plotting web with lambda threshold {thresh}')
        voids = data_slice[data_slice['l1'] < thresh]
        sheet = data_slice[(data_slice['l2'] < thresh) & (data_slice['l1'] > thresh)]
        filam = data_slice[(data_slice['l2'] > thresh) & (data_slice['l3'] < thresh)]
        knots = data_slice[data_slice['l3'] > thresh]
        data_slice['env'] = data_slice[['l1', 'l2', 'l3']].apply(lambda x: find_env(*x, thresh), axis=1)

    else:
        print(f'Plotting web with pre-computed environment class')
        ind_voids = np.where(ordered_envs == 0)
        ind_sheet = np.where(ordered_envs == 1)
        ind_filam = np.where(ordered_envs == 2)
        ind_knots = np.where(ordered_envs == 3)
        
        #print(ind_voids, ind_sheet, ind_filam, ind_knots[0])

        voids = data_slice[data_slice['env'] == ind_voids[0][0]]
        sheet = data_slice[data_slice['env'] == ind_sheet[0][0]]
        filam = data_slice[data_slice['env'] == ind_filam[0][0]]
        knots = data_slice[data_slice['env'] == ind_knots[0][0]]

    n_pts = float(len(data))
    #str_env_types = '%.2f & %.2f & %.2f & %.2f \\\ ' % (len(voids)/n_pts, len(sheet)/n_pts, len(filam)/n_pts, len(knots)/n_pts)
    #print(str_env_types)
    str_dens_types = '%.2f & %.2f & %.2f & %.2f \\\ ' % (voids['dens'].median(), sheet['dens'].median(), filam['dens'].median(), knots['dens'].median())
    #print(str_dens_types)

    if do_plot:

        fontsize = 25

        if grid == 32:
            size = 40
        elif grid == 64:
            size = 15
        elif grid == 128:
            size = 30
        elif grid == 256:
            size = 0.1

        print(f'Point size={size}')

        # Plot the eigenvaule threshold based V-Web
        plt.figure(figsize=(10, 10))
        plt.rcParams["axes.edgecolor"] = "0.0"
        plt.rcParams["axes.linewidth"]  = 1
        plt.xlim([-shift, shift])
        plt.ylim([-shift, shift])

        plt.xlabel(r'SGX $\quad [h^{-1} Mpc]$', fontsize=fontsize)
        plt.ylabel(r'SGY $\quad [h^{-1} Mpc]$', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        #if use_thresh:
        if title != None:
            plt.title(title, fontsize=fontsize)

        print('voids: ', len(voids))
        print('sheet: ', len(sheet))
        print('filam: ', len(filam))
        print('knots: ', len(knots))

        plt.scatter(voids['x'], voids['y'], c='lightgrey', s=size, marker='s')
        plt.scatter(sheet['x'], sheet['y'], c='grey', s=size, marker='s')
        plt.scatter(filam['x'], filam['y'], c='black', s=size, marker='s')
        plt.scatter(knots['x'], knots['y'], c='red', s=size, marker='s')
        plt.rcParams["axes.edgecolor"] = "0.0"
        plt.rcParams["axes.linewidth"]  = 1

        # Save file
        f_out = fout + '_' + str(thresh).replace('.','') + '.png'
        plt.tight_layout()
        print('Saving fig to ', f_out)
        plt.savefig(f_out)
        plt.cla()
        plt.clf()

        # Plt densities
        if plot_dens:
            #palette="YlOrBr"
            #palette="PuRd"
            palette="Greys"
            color_fac = 50.0
            #size = 100
            plt.figure(figsize=(10, 10))
            plt.xlim([-shift, shift])
            plt.ylim([-shift, shift])
            plt.title('$\log_{10}\Delta_m', fontsize=fontsize)
            #sns.scatterplot(data['x'], data['y'], hue=np.log10(10 * data['dens']), marker='s', s=size, legend = False, palette=palette)
            plt.scatter(data['x'], data['y'], c=color_fac * np.log10(data['dens']), marker='s', s=size, cmap=palette)

            # Override seaborn defaults
            plt.xlabel(r'SGX $\quad [h^{-1} Mpc]$', fontsize=fontsize)
            plt.ylabel(r'SGY $\quad [h^{-1} Mpc]$', fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.tight_layout()

            # Save file
            f_out = fout + '_dens.png'
            print('Saving fig to ', f_out)
            plt.savefig(f_out)
            plt.cla()
            plt.clf()

    return data_slice


def check_distribution(values):
    """ Quick sanity check on a distribution """

    std = np.std(values)
    med = np.median(values)
    mea = np.mean(values)

    print(f'Mean: {mea}, median: {med}, stddev: {std}')


def entropy(labels=None, k=None):
    """ Very simple entropy calculation """

    n = len(labels)

    e_i = 0.0
    for i_k in range(0, k):
        n_i = len(np.where(labels == i_k)[0])
        e_i -= (n_i / n) * np.log(n_i / n)

    print('Entropy : ', e_i)
    return e_i


@t.time_total
def kmeans_stability(data=None, n_clusters_max=None, n_ks=10, rescale_factor=1000, verbose=False, f_out=None):
    """ Check how stable are the centers of the k-means, comparing for different ks """

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

            if verbose == True:
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

    if f_out != None:
        for i_col, col in enumerate(columns):
            df_print[col] = results[:, i_col]
        
        sns.lineplot(x=df_print[columns[0]], y=df_print[columns[1]])
        sns.lineplot(x=df_print[columns[0]], y=df_print[columns[4]])
        plt.legend(labels=['Center Scatter', 'Center Scatter Norm.'])
        plt.show()


@t.time_total
def plot_local_volume_density_slice(data=None, box=100, file_out=None, title=None, envirs=None):
    """
    Plot a slice of the local volume with a different color coding for each environment
    Make sure the environments are correctly sorted according to their density
    """

    z_min = box * 0.5 - thick
    z_max = box * 0.5 + thick

    data = data[data['z'] > z_min]
    data = data[data['z'] < z_max]

    #data['env_name'] = data['env'].apply(lambda x: envirs_sort[x])
    data['x'] = data['x'].apply(lambda x: x - 50.0)
    data['y'] = data['y'].apply(lambda x: x - 50.0)

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
        tmp_env = data[data['env_sort'] == env]
        plt.scatter(tmp_env['x'], tmp_env['y'], c=colors[ie], s=size, marker='s')

    plt.xlabel(r'SGX $\quad [h^{-1} Mpc]$', fontsize=fontsize)
    plt.ylabel(r'SGY $\quad [h^{-1} Mpc]$', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(file_out)


@t.time_total
def plot_lambda_distribution(data=None, grid=128, base_out=None, env_col='env', envirs=[0., 1., 2., 3.], cols=['l1', 'l2', 'l3'], x_axis=True):
    """
    Plot the distributions of the different eigenvalues in each environment type
    Make sure the environments are correctly sorted according to their density
    """
    
    print('Plotting lambda distributions...')
    labels = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$']
    #x_label = r'$\lambda _1, \lambda _2, \lambda _3$'
    x_label = r'$\lambda$'
    colors = ['blue', 'green', 'grey']
    fontsize = 20

    # First loop over different environments 
    for ie, env in enumerate(envirs):

        # Set plot settings
        fig, ax = plt.subplots()
        
        if x_axis:
            plt.figure(figsize=(6, 6))
        else:
            plt.figure(figsize=(6, 5))

        plt.grid(False)

        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        print('Environment: ', env)
        tmp_env = data[data['env'] == env]
        env_str = str(env)
        file_out = base_out + '_env' + env_str + '.png'
        
        # Now plot the eigenvalue distribution in each environment
        for il, col in enumerate(cols):
            plt.hist(tmp_env[col].values, bins=50, color=colors[il], label=labels[il], alpha=0.7, density=True)

        if grid == 32:
            plt.xlim([-0.5, 0.5])
        elif grid == 64:
            plt.xlim([-1, 1.0])
        elif grid == 128:
            plt.xlim([-1, 3.0])

        xticks = plt.xticks()[0]
        yticks = plt.yticks()[0] 
        xticks_str = ['%.1f' % x for x in xticks]
        yticks_str = ['%.1f' % y for y in yticks]
        plt.yticks(ticks=yticks, labels=yticks_str)

        if x_axis:
            plt.xlabel(x_label, fontsize=fontsize)
            plt.xticks(ticks=xticks, labels=xticks_str)
        else:
            plt.xticks(ticks=[], labels=[])

        plt.tight_layout()
    
        if env == 0.0:
            plt.legend(prop={'size':fontsize})

        # Save figure and clean plot
        print(f'Plotting lambda distribution to: {file_out}')
        plt.savefig(file_out)
        plt.clf()
        plt.cla()
        plt.close()


if __name__ == "__main__":
    """ Main program, used for debugging and testing """

    assign_halos_to_environment_type()

    pass



