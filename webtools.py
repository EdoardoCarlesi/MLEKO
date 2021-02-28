"""
    MLEKO
    Machine Learning Ecosystem for KOsmology

    (C) Edoardo Carlesi 2020
    https://github.com/EdoardoCarlesi/MLEKO
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tools as t

from sklearn.metrics import silhouette_score, homogeneity_score, calinski_harabasz_score, silhouette_samples
from yellowbrick.cluster import KElbowVisualizer
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

    
def plot_eigenvalues_per_environment_type(data=None, env_type=None, out_base=None, grid=None): 
    """
        Plot the three eigenvalues distributions for a given environment type
    """

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


def plot_new_format(data=None, f_out=None, labels=None):
    """
        Plot the point distribution according to Yehuda's new suggestion
    """

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

    cols = []
        
    for il, cl in enumerate(labels):
        cols.append(colors_sort[cl])

    print('Plotting in 3D particle distribution...')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data['l1'], data['l2'], data['l3'], c = cols)
    ax.set_xlabel(r'$\lambda_1$')
    ax.set_ylabel(r'$\lambda_2$')
    ax.set_zlabel(r'$\lambda_3$')
    plt.tight_layout()
    plt.savefig(f_out)
    plt.clf()
    plt.cla()
    print('Done.')


def wss(data=None, centers=None, labels=None):
    """
        Within sum of squares
    """
    
    '''
    for i, center in enumerate(centers):
        for 
    '''


@t.time_total
def generate_random(data=None, grid=None, prior='flat', verbose=False, mode='simple'):
    """
        Generate a random distribution of eigenvalues
    """

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
def evaluate_metrics(data=None, n_clusters_max=None, n_init=10, rescale_factor=10, visualize=False, elbow=True):
    """
        Evaluate different metrics to estimate the best k for the cluster number
    """

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
        visualizer = KElbowVisualizer(model, k=(2, n_clusters_max), timings=False)

        if visualize:
            visualizer.fit(X)        
            visualizer.show()
            plt.show()

        print('Elbow Method score for Calinski-Harabasz...')
        visualizer = KElbowVisualizer(model, k=(2, n_clusters_max), timings=False, metric='calinski_harabasz', locate_elbow=True)

        if visualize:
            visualizer.fit(X)        
            visualizer.show()
            plt.show()

    return kmeans


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
        #print(str_env)

    #print('\hline')

    return data


def compare_vweb_kmeans(vweb=None, l=0.0):
    """ Compare the fraction of volume occupied by different environmental types in the usual vweb classification and in kmeans """

    shared = []
    n_env_shared = np.zeros(4)
    tot_env = np.zeros(4)

    diffs = vweb['env'].values - vweb['envk'].values
    diffs_inds = np.where(diffs == 0)

    n_all = len(vweb)
    print(f'Global shared values: {l} {len(diffs[diffs_inds])/n_all}')
    
    for i, env in enumerate([0, 1, 2, 3]):
        n_tot = len(vweb[vweb['envk'] == env])
        tmp = vweb[vweb['env'] == env]
        n_shared = len(tmp[tmp['envk'] == env])
        
        n_env_shared[i] = n_shared
        tot_env[i] = n_tot

        #print(f'Env: {env}, Tot: {n_tot}, Shared: {n_shared}, Perc: {n_env_shared/tot_env}')

    average = np.mean(n_env_shared/tot_env)
    print(f'Global shared values: {l} {len(diffs[diffs_inds])/n_all} {average} {n_env_shared/tot_env}')

    for env in [0, 1, 2, 3]:
        n_tot = len(vweb[vweb['env'] == env])
        tmp = vweb[vweb['envk'] == env]
        n_shared = len(tmp[tmp['env'] == env])
        
        #print(f'(inverse check) Env: {env}, Tot: {n_tot}, Shared: {n_shared} Perc: {n_shared/n_tot}')


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
def plot_vweb(data=None, fout=None, thresh=0.0, grid=64, box=100.0, thick=2.0):
    """
        Plot the usual vweb using an input threshold and a given dataset
    """

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

    # Plot the eigenvaule threshold based V-Web
    print(f'Plotting web with lambda threshold {thresh}')

    plt.figure(figsize=(10, 10))
    plt.xlim([-shift, shift])
    plt.ylim([-shift, shift])
    plt.title('$\lambda_{thr} = $' + str(thresh), fontsize=fontsize)
    plt.xlabel(r'SGX $\quad [h^{-1} Mpc]$', fontsize=fontsize)
    plt.ylabel(r'SGY $\quad [h^{-1} Mpc]$', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    size = 30
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


def check_distribution(values):
    """
        Quick sanity check on a distribution
    """

    std = np.std(values)
    med = np.median(values)
    mea = np.mean(values)

    print(f'Mean: {mea}, median: {med}, stddev: {std}')


def entropy(labels=None, k=None):
    """
        Very simple entropy calculation
    """

    n = len(labels)

    e_i = 0.0
    for i_k in range(0, k):
        n_i = len(np.where(labels == i_k)[0])
        e_i -= (n_i / n) * np.log(n_i / n)

    print('Entropy : ', e_i)
    return e_i


@t.time_total
def kmeans_stability(data=None, n_clusters_max=None, n_ks=10, rescale_factor=1000, verbose=False, f_out=None):
    """
        Check how stable are the centers of the k-means, comparing for different ks
    """

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

    print(results)

    if f_out != None:
        for i_col, col in enumerate(columns):
            df_print[col] = results[:, i_col]
        
        print(df_print.head())

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
def plot_lambda_distribution(data=None, grid=None, base_out=None, envirs=None):
    """
        Plot the distributions of the different eigenvalues in each environment type
        Make sure the environments are correctly sorted according to their density
    """

    labels = ['$\lambda_1$', '$\lambda_2$', '$\lambda_3$']

    for il, col in enumerate(cols_select):

        for ie, env in enumerate(envirs):
            tmp_env = data[data['env_name'] == env]
            sns.distplot(tmp_env[col], color=colors[ie], label=env)

        env_str = envirs_sort[i]
        file_out = base_out + str_grid + col + '.png'

        print(f'Plotting lambda distribution to: {file_out}')
        fontsize=10

        # Plot the three eigenvalues
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        if grid == 32:
            plt.xlim([-0.5, 0.5])
        elif grid == 64:
            plt.xlim([-1, 1.0])
        elif grid == 128:
            plt.xlim([-1, 3.0])

        plt.xlabel(labels[il], fontsize=fontsize)
        plt.legend()
        plt.tight_layout()
    
        # Save figure and clean plot
        plt.savefig(file_out)
        plt.clf()
        plt.cla()


if __name__ == "__main__":
    """
        Main program, used for debugging and testing
    """





