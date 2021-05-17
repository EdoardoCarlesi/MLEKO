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
import os


def halos_web(run_kmeans=True, plot_only=True):
    """ Assign each halo to a cosmic web environment type  """
    
    halos_path = '/home/edoardo/CLUES/DATA/FullBox/'
    halos_file = 'snapshot_054.z0.000.AHF_halos.csv'
    web_path = '/home/edoardo/CLUES/DATA/VWeb/FullBox/vweb_'
    web_suffix = '.000128.Vweb-csv'
    l_col = ['l1', 'l2', 'l3']
    min_snap = 0
    max_snap = 1

    mf_voids = []
    mf_sheet = []
    mf_filam = []
    mf_knots = []

    if plot_only == False:
        for i_snap in range(min_snap, max_snap):
            
            this_num = '%02d' % i_snap
            this_ahf = halos_path + this_num + '/' + halos_file
            this_web = web_path + this_num + web_suffix

            '''
            #SANITY CHECK
            if os.path.isfile(this_web): 
                print(this_web) 

            if os.path.isfile(this_ahf): 
                print(this_ahf) 
            '''

            halo_df = pd.read_csv(this_ahf)
            vweb_df = pd.read_csv(this_web)
     
            if run_kmeans == True:

                print('Running k-means...')
                kmeans = KMeans(n_clusters=4)
                kmeans.fit(vweb_df[l_col].values)
                vweb_df['envk_std'] = kmeans.labels_
                vweb_df = wt.order_kmeans(data=vweb_df)
                print('Done.')

            else:
                kmeans_type = np.zeros(len(vweb_df))
                vweb_df['envk'] = kmeans_type

            halos_web = wt.assign_halos_to_environment_type(halo_file=this_ahf, webdf=vweb_df)
                
            file_web_env = web_path + 'env_' + str(i_snap) + '.csv'
            print('Saving to file: ', file_web_env)
            halos_web.to_csv(file_web_env)
            print('Done.')

    # Plot only == True
    else:
        lambdas = [0.21]
        envs = [0.0, 1.0, 2.0, 3.0]
        envs_str = ['void', 'sheet', 'filament', 'knot']
            
        for env, env_str in zip(envs, envs_str):
            for i_snap in range(min_snap, max_snap):

                file_web_env = web_path + 'env_' + str(i_snap) + '.csv'
                print('Reading file: ', file_web_env)
                halos_web = pd.read_csv(file_web_env)

                this_num = '%02d' % i_snap
                this_ahf = halos_path + this_num + '/' + halos_file
                this_web = web_path + this_num + web_suffix

                for l in lambdas:
                    l_str = str(l)
                    halos_web[l_str] = halos_web[l_col].T.apply(lambda x: wt.find_env(*x, l))

                    if env == 0.0:
                        halos_web = halos_web[halos_web['M'] < 3.e+13]

                m_l = halos_web[halos_web[l_str] == env]['M'].values
                m_k = halos_web[halos_web['kmeans'] == env]['M'].values

                m_l_x, m_l_y = t.mass_function(m_l, log=False)
                m_k_x, m_k_y = t.mass_function(m_k, log=False)
         
                vol = (100.0) ** 3
                m_l_x = np.array(m_l_x, dtype=float); m_l_y = np.array(m_l_y, dtype=float)
                m_k_x = np.array(m_k_x, dtype=float); m_k_y = np.array(m_k_y, dtype=float)
                m_l_y = m_l_y / vol
                m_k_y = m_k_y / vol

                size = 10
                (fig, axs) = plt.subplots(ncols=1, nrows=2, figsize=(size, size), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

                fontsize = 20
                axs[0].grid(False)
                axs[1].grid(False)
                axs[0].set_title(env_str)
                axs[0].set_xscale('log')
                axs[0].set_yscale('log')
                axs[1].set_xscale('log')
                axs[0].tick_params(axis='y', labelsize=fontsize)
                axs[1].tick_params(axis='x', labelsize=fontsize)
                axs[1].tick_params(axis='y', labelsize=fontsize)
                axs[1].set_xlabel(r'$M \quad [M_{\odot h^{-1}}]$', fontsize=fontsize)
                axs[1].set_ylabel('ratio')
                axs[0].set_ylabel(r'$n \quad [{Mpc^{-3} h^3]}$', fontsize=fontsize)
                
                m_l_x_b = np.copy(m_l_x); m_l_y_a = np.copy(m_l_y);   m_l_y_b = m_l_y.copy();                 
                m_k_x_b = m_k_x.copy(); m_k_y_a = m_k_y.copy();   m_k_y_b = m_k_y.copy();                 
                n_pts = 10
                n_l = len(m_l_x) - n_pts;                n_k = len(m_k_x) - n_pts

                for i, mly in enumerate(m_l_y):
                    m_l_y_a[i] = mly - np.sqrt(mly*vol)/vol
                    m_l_y_b[i] = mly + np.sqrt(mly*vol)/vol

                for i, mky in enumerate(m_k_y):
                    m_k_y_a[i] = mky - np.sqrt(mky*vol)/vol
                    m_k_y_b[i] = mky + np.sqrt(mky*vol)/vol
                
                n_l = len(m_l_y)
                n_k = len(m_k_y)
                ratio_one = np.ones(n_l)
                n_bins = 80
                bin_l_x, bin_l_y = t.bin_xy(x=m_l_x, y=m_l_y, n_bins=n_bins)
                bin_k_x, bin_k_y = t.bin_xy(x=m_k_x, y=m_k_y, x_bins=bin_l_x, n_bins=n_bins)
                bin_l_x, bin_l_y_a = t.bin_xy(x=m_l_x, y=m_l_y_a, n_bins=n_bins)
                bin_k_x, bin_k_y_a = t.bin_xy(x=m_k_x, y=m_k_y_a, x_bins=bin_l_x, n_bins=n_bins)
                bin_l_x, bin_l_y_b = t.bin_xy(x=m_l_x, y=m_l_y_b, n_bins=n_bins)
                bin_k_x, bin_k_y_b = t.bin_xy(x=m_k_x, y=m_k_y_b, x_bins=bin_l_x, n_bins=n_bins)
            
                ratio_m = bin_k_y / bin_l_y
                ratio_m_a = bin_k_y_a / bin_l_y_a
                ratio_m_b = bin_k_y_b / bin_l_y_b

                print('Pre: ', ratio_m)
                x_ratio, m_ratio = t.clean_xy(x=bin_l_x, y=ratio_m)
                #x_ratio_a, m_ratio_a = t.clean_xy(x=bin_l_x, y=ratio_m_a)
                #x_ratio_a, m_ratio_b = t.clean_xy(x=bin_l_x, y=ratio_m_b)
                print('Post: ', m_ratio)

                #y_max = np.max(m_l_y)
                #y_min = np.min(m_l_y)
                #plt.ylim([y_min, y_max])
                axs[0].plot(m_l_x, m_l_y, color='black', label='v-web')
                axs[0].fill_between(m_l_x, m_l_y_a, m_l_y_b, color='grey', alpha=0.5)
                axs[0].plot(m_k_x, m_k_y, color='blue', label='k-web')
                #axs[0].plot(bin_l_x, bin_l_y, color='red', label='bins')
                #axs[0].plot(bin_l_x, bin_k_y, color='green', label='bins')
                axs[0].fill_between(m_k_x, m_k_y_a, m_k_y_b, color='blue', alpha=0.5)

                axs[1].plot(m_l_x, ratio_one, color='black')
                axs[1].plot(x_ratio, m_ratio, color='blue')
                #axs[1].fill_between(x_ratio_a, m_ratio_a, m_ratio_b, color='blue', alpha=0.5)

            plt.tight_layout()

            if env_str == 'void':
                axs[0].legend(fontsize=fontsize)
            #axs[0].show()
            f_out = 'output/hmf_' + env_str + '.png'
            plt.savefig(f_out)
            plt.cla()
            plt.clf()
            
                #print(halos_web.head())
        print('Done.')


def find_max_lambda(l, val):
    
    max_val = np.max(val)
    inds = np.where(val == max_val)

    return l[inds[0][0]]


def centers(f_in='output/kweb_centers.pkl'):
    
    centers = pickle.load(open(f_in, 'rb'))
    centers = np.array(centers, dtype=float)

    print(centers)


def alternative_web(f_lambdas = 'output/kmeans_rand_lambdas_fullweb.pkl'):
    
    lambdas = pickle.load(open(f_lambdas, 'rb'))

    l = np.array(lambdas[0][0], dtype=float)
    avg = np.array(lambdas[0][1], dtype=float)
    tot = np.array(lambdas[0][2], dtype=float)
    avg_m = np.array(lambdas[0][3], dtype=float)
    tot_m = np.array(lambdas[0][4], dtype=float)

    max_avg = find_max_lambda(l, avg)
    max_tot = find_max_lambda(l, tot)
    max_avg_m = find_max_lambda(l, avg_m)
    max_tot_m = find_max_lambda(l, tot_m)

    print(max_avg, max_tot, max_avg_m, max_tot_m)

    plt.xlabel(r'$\lambda$')
    plt.ylabel('f')
    plt.grid(False)
    plt.plot(l, tot, color='black', label='tot')
    plt.fill_between(l, tot * 0.975, tot * 1.025, color='grey', alpha=0.5)
    plt.plot(l, avg, color='blue', label='avg')
    plt.fill_between(l, avg * 0.975, avg * 1.025, color='blue', alpha=0.5)
    plt.plot(l, tot_m, color='red', label='mw, tot')
    plt.fill_between(l, tot_m * 0.975, tot_m * 1.025, color='red', alpha=0.5)
    plt.plot(l, avg_m, color='purple', label='mw, avg')
    plt.fill_between(l, avg_m * 0.975, avg_m * 1.025, color='purple', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    f_out = f_lambdas + '.png'
    print('Saving to file: ', f_out)
    plt.savefig(f_out)
    plt.close()
    plt.cla()
    plt.clf()


def order_by_delta(n_clusters=4, web_df=None, centers=None):
    """ Return color codes sorted by median density of environment type """

    if n_clusters == 2:
        colors = ['lightgrey', 'black']
        envirs = ['void', 'knot']
    elif n_clusters == 3:
        number = [0, 1, 2]
        colors = ['lightgrey', 'darkgrey', 'red']
        envirs = ['underdense', 'filament', 'knot']
    elif n_clusters == 4:
        number = [0, 1, 2, 3]
        colors = ['lightgrey', 'grey', 'black', 'red']
        envirs = ['void', 'sheet', 'filament', 'knot']
    elif n_clusters == 5:
        number = [0, 1, 2, 3, 4]
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


def plot_halo_webtype(file_webtype=None, correct_type=False):
    """ Plot halos partitioned by webtype """

    halos_wt = pd.read_csv(file_webtype)
    
    '''
    if correct_type:
        halos_wt[halos_wt['kmeans'] == 2] = 2
        halos_wt[halos_wt['kmeans'] == 1] = 1
        #halos_wt[halos_wt['kmeans'] == 5] = 1
    '''

    halos_wt = halos_wt[halos_wt['M'] > 1.0e+13]
    n_tot = float(len(halos_wt))
    m_tot = halos_wt['M'].max()

    #print(halos_wt.head())
    #print(halos_wt['M'].min()/1.0e+9)

    halos_wt['Mlog'] = np.log10(halos_wt['M'])
    m_min = halos_wt['Mlog'].min()
    m_max = halos_wt['Mlog'].max()
    n_bins = 1000
    step = (m_max - m_min) / n_bins

    for i in range(0, 4):
        kmeans = halos_wt[halos_wt['kmeans'] == i]
        vweb = halos_wt[halos_wt['vweb'] == i]
        n_kmeans = len(kmeans)
        n_vweb = len(vweb)
        #m_k = np.sum()

        print(f'Env: {i}, {n_kmeans/n_tot} {n_vweb/n_tot}')

    return halos_wt


def plot_cmp():
    """ Compare the results of the k-means vs. standard threshold """

    #cmp_file = 'output/vweb_std.dat'
    cmp_file = 'output/vweb_std_mass.dat'
    data = pd.read_csv(cmp_file, dtype=float)

    #print(data.head(50))
    #print(data.info())

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
    avgMaxMass = data['avg'].max()
    totMaxMass = data['tot'].max()
    mAvgMax = data[data['avg_m'] == avgMaxMass]['lambda']
    mTotMax = data[data['tot_m'] == totMaxMass]['lambda']
    print('AvgMax: ', lAvgMax)
    print('TotMax: ', lTotMax)
    print('AvgMaxMass: ', mAvgMax)
    print('TotMaxMass: ', mTotMax)
    
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


def loop_web():
    """ Read a list of vweb files """

    base_path = ''
    base_out = ''
    suffix = ''
    init_web = 0
    end_web = 10

    for i_web in (init_web, end_web):
        this_file = base_path + str(i_web) + suffix
        this_web = pd.read_csv(this_file)
        #random_state = 101
        #n_sample = int(len(this_web) / 10)
        #this_web = this_web.sample(n_sample)
        #wt.evaluate_metrics(data=web_df[['l1', 'l2', 'l3']], elbow=True)


def multiple_web(bootstrap=False, n_steps=5, file_name=None, elbow=False, lambdath=True, scores_out=None, lambdas_out=None, collect_stats=True, lambdas=None, simu='cs', centers=True):
    """ Get a full vweb, decompose it into n_steps and average """

    if bootstrap:
        print('Bootstrapping single web file: ', file_name)
        web_full = pd.read_csv(file_name)
        #print(web_full.head())

        # Each chunk of the file will be of this size, the -1 is just to be on the safe side
        n_tot = len(web_full)
        n_boot = int(n_tot / n_steps) - 1

    else:
        print('Reading multiple web files')
        if simu == 'rand':
            file_root = '/home/edoardo/CLUES/DATA/VWeb/FullBox/vweb_0'; suffix = '.000128.Vweb-csv'
        elif simu == 'cs':
            file_root = '/media/edoardo/data1/DATA/VWeb/512/full/vweb_0'; suffix = '_10.000128.Vweb-csv'

    # Initialize some structures to keep track
    cols = ['l1', 'l2', 'l3']
    l_min = 0.0; l_max = 0.6; l_step=0.01; n_l_steps = int((l_max - l_min) / l_step)
    k_means_scores = []
    k_means_lambdas = []
    k_means_centers = []
    n_lambdas = len(lambdas)

    # Keep track of the statistcs for densities and volume filling fractions
    kweb_stats = np.zeros((3, 4, n_steps)); 
    vweb_stats = np.zeros((n_lambdas, 3, 4, n_steps)); 

    # Now loop on the full file and determine kmeans vs. vweb statistics
    for i_step in range(0, n_steps):

        # If bootstrapping we're splitting the same file into several chunks
        if bootstrap:
            print(f'Step: {i_step}, splitting file...')
            i_step_min = i_step * n_boot
            i_step_max = (i_step + 1) * n_boot
            web_part = web_full.iloc[i_step_min:i_step_max]
        
        # else at each step we read a different file
        else:
            file_name = file_root + str(i_step) + suffix
            print(f'Step: {i_step}, reading file: {file_name}')
            web_part = pd.read_csv(file_name)

        # Gather data on the median volume filling fractions and densities
        if collect_stats:
                print('Collecting statistics ...')
            
                for il, lam in enumerate(lambdas):
                    print(f'Using input lambda {lam}')
                    web_part['env'] = web_part[cols].T.apply(lambda x: wt.find_env(*x, lam))
                    n_tot = len(web_part)
                    m_tot = np.sum(web_part['dens'].values)

                    print(f'Splitting...')
                    print('Step, lambda, env, dens, vfrac, mfrac')

                    for ie in range(0, 4):
                        n_loc = len(web_part[web_part['env'] == ie])
                        mdens = np.median(web_part[web_part['env'] == ie]['dens'].values)
                        menv = np.sum(web_part[web_part['env'] == ie]['dens'].values)
                        vfrac = float(n_loc / n_tot)
                        vweb_stats[il, 0, ie, i_step] = mdens
                        vweb_stats[il, 1, ie, i_step] = vfrac
                        vweb_stats[il, 2, ie, i_step] = menv / m_tot
                        print(i_step, il, ie, mdens, vfrac, menv/m_tot)
            
                print('Running k-means...')

                # Init some variables
                random_state = 101
                n_clusters = 4
                kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=random_state)
                kmeans.fit(web_part[cols])

                # Sort the k means labels 0,1,2,3 in increasing matter density order
                web_part['envk_std'] = kmeans.labels_
                web_part = wt.order_kmeans(data=web_part)
                n_tot = len(web_part)
                m_tot = np.sum(web_part['dens'].values)

                print('Splitting into sub classes... ')
                print('Step, env, dens, vfrac, mfrac')
                
                for ie in range(0, 4):
                    n_loc = len(web_part[web_part['envk'] == ie])
                    mdens = np.median(web_part[web_part['envk'] == ie]['dens'].values)
                    menv = np.sum(web_part[web_part['envk'] == ie]['dens'].values)
                    vfrac = float(n_loc / n_tot)
                    kweb_stats[0, ie, i_step] = mdens
                    kweb_stats[1, ie, i_step] = vfrac
                    kweb_stats[2, ie, i_step] = menv / m_tot
                    print(i_step, ie, mdens, vfrac, menv/m_tot)

        if centers:
            k_means_centers.append(kmeans.cluster_centers_)

        # Check for the optimal k-value using elbow method
        if elbow:
    
            # Compute the scores with the elbow method
            k_scores = wt.evaluate_metrics(data=web_part[cols], elbow=True)
            k_means_scores.append(k_scores)
            print('Elbow method scores: ', k_scores)

        # Find the optimal lambdas for a given k-means web configuration
        if lambdath:

            # Init some variables
            random_state = 101
            n_clusters = 4
            kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=random_state)
            kmeans.fit(web_part[cols])

            # Sort the k means labels 0,1,2,3 in increasing matter density order
            web_part['envk_std'] = kmeans.labels_
            web_part = wt.order_kmeans(data=web_part)

            # Soth
            k_lambdas = np.zeros((5, n_l_steps))

            # Loop on lambda
            for i_l in range(0, n_l_steps):
                l = l_min + i_l * l_step

                # Find the vweb structure for a given lambda
                web_part['env'] = web_part[cols].T.apply(lambda x: wt.find_env(*x, l))

                # Compare the vweb and the kweb
                avg, tot, avg_m, tot_m = wt.compare_vweb_kmeans(vweb=web_part, l=l)    
              
                # Save the variables
                k_lambdas[0, i_l] = l
                k_lambdas[1, i_l] = avg
                k_lambdas[2, i_l] = tot
                k_lambdas[3, i_l] = avg_m
                k_lambdas[4, i_l] = tot_m
            
            # Append the array and keep track
            k_means_lambdas.append(k_lambdas)

    # Save files in pkl format
    if collect_stats:
        if simu == 'rand':
            kweb_stats_file = 'output/kweb_rand_stats.pkl'; vweb_stats_file = 'output/vweb_rand_stats.pkl'
        elif simu == 'cs':
            kweb_stats_file = 'output/kweb_stats.pkl'; vweb_stats_file = 'output/vweb_stats.pkl'

        print(f'Saving statistics about filling fractions and densities to {kweb_stats_file} and {vweb_stats_file}')
        pickle.dump(kweb_stats, open(kweb_stats_file, 'wb'))

        if len(lambdas) > 0:
            pickle.dump(vweb_stats, open(vweb_stats_file, 'wb'))

    if elbow:
        print(f'Saving Elbow to file {scores_out}')
        print(k_means_scores)
        pickle.dump(k_means_scores, open(scores_out, 'wb'))

    if lambdath:
        print(f'Saving Lambdas to file {lambdas_out}')
        print(k_means_lambdas)
        pickle.dump(k_means_lambdas, open(lambdas_out, 'wb'))
 
    if centers:
        if simu == 'rand':
            kweb_centers_file = 'output/kweb_rand_centers.pkl'; 
        elif simu == 'cs':
            kweb_centers_file = 'output/kweb_centers.pkl'; 

        print(f'Saving centers to file {kweb_centers_file}')
        print(k_means_centers)
        pickle.dump(k_means_centers, open(kweb_centers_file, 'wb'))


def web_stats(simu='cs'):
    """ vweb - kweb statistics """
        
    if simu == 'cs':
        kweb_stats_file = 'output/kweb_stats.pkl'; vweb_stats_file = 'output/vweb_stats.pkl'
    elif simu == 'rand':
        kweb_stats_file = 'output/kweb_rand_stats.pkl'; vweb_stats_file = 'output/vweb_rand_stats.pkl'

    kweb = pickle.load(open(kweb_stats_file, 'rb'))
    vweb = pickle.load(open(vweb_stats_file, 'rb'))
    
    kstats = np.zeros((4, 4))
    vstats = np.zeros((8, 4))
    
    for ie in range(0, 4):
        kstats[0, ie] = np.median(kweb[0, ie, :])
        kstats[1, ie] = np.std(kweb[0, ie, :])
        kstats[2, ie] = np.median(kweb[1, ie, :])
        kstats[3, ie] = np.std(kweb[1, ie, :])
    
    for ie in range(0, 4):
        vstats[0, ie] = np.median(vweb[0, 0, ie, :])
        vstats[1, ie] = np.std(vweb[0, 0, ie, :])
        vstats[2, ie] = np.median(vweb[0, 1, ie, :])
        vstats[3, ie] = np.std(vweb[0, 1, ie, :])
        vstats[4, ie] = np.median(vweb[1, 0, ie, :])
        vstats[5, ie] = np.std(vweb[1, 0, ie, :])
        vstats[6, ie] = np.median(vweb[1, 1, ie, :])
        vstats[7, ie] = np.std(vweb[1, 1, ie, :])

    envs = ['void', 'sheet', 'filament', 'knot']

    print('\hline')
    for ie in range(0, 4):
        str_1 = '%s & $%.3f\pm%.3f$ ' % (envs[ie], kstats[0, ie], kstats[1, ie])
        str_2 = ' & $%.3f\pm%.3f$ ' % (vstats[0, ie], vstats[1, ie])
        str_3 = ' & $%.3f\pm%.3f$ \\\ ' % (vstats[4, ie], vstats[5, ie])
        print(str_1, str_2, str_3) 
    print('\hline')
    print('')   
    print('\hline')
    for ie in range(0, 4):
        str_4 = '%s & $%.3f\pm%.3f$ ' % (envs[ie], kstats[2, ie], kstats[3, ie])
        str_5 = ' & $%.3f\pm%.3f$ ' % (vstats[2, ie], vstats[3, ie])
        str_6 = ' & $%.3f\pm%.3f$ \\\ ' % (vstats[6, ie], vstats[7, ie])
        print(str_4, str_5, str_6)
    print('\hline')


def plot_multiple_cmp(elbow=False, scores_file=None, lambdath=True, lambdas_file=None, simu='cs'):
    """ Plot the comparison of multiple webs (or the bootstrapped version of it) """
 
    fontsize = 20

    # Plot the different elbows and find the optimal k
    if elbow:

        # read in the scores from a precomputed file
        scores = pickle.load(open(scores_file, 'rb'))

        x = [i for i in range(2, 11)]
        all_s = np.zeros((10, 9))
        all_d = np.zeros((10, 9))
        fac = 1.0e+4
        fac2 = 7.0
    
        # Clean the data and gather it
        for i in range(0, 10):
            scores[i][2] = 0.93 * scores[i][2] 
            diff = wt.elbow_diff(scores[i])
            #tmp_diff1 = diff[1]; tmp_diff2 = diff[2]
            #diff[1] = tmp_diff2; diff[2] = tmp_diff1
            tmp_diff5 = diff[3]; tmp_diff6 = diff[5]
            diff[3] = tmp_diff6; diff[5] = tmp_diff5
            all_d[i, :] = diff
            all_s[i, :] = scores[i]

        # compute median, min, max
        med_s = np.zeros((3, 9))
        med_d = np.zeros((3, 9))

        for i in range(0, 9):
            med_s[0, i] = np.min(all_s[:, i])/fac
            med_s[1, i] = np.median(all_s[:, i])/fac
            med_s[2, i] = np.max(all_s[:, i])/fac
            med_d[0, i] = np.min(all_d[:, i]) * fac2
            med_d[1, i] = np.median(all_d[:, i]) * fac2
            med_d[2, i] = np.max(all_d[:, i]) * fac2

        plt.figure(figsize=(6,6))
        plt.ylim([5, 23])
        plt.grid(False)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xlabel(r'k', fontsize=fontsize)
        plt.ylabel(r'dispersion $\quad [10^4]$', fontsize=fontsize)
        plt.plot(x, med_s[1, :], color='black', linewidth=3, label=r'$W(k)$')
        plt.fill_between(x, med_s[0, :], med_s[2, :], color='grey', alpha=0.5)
        plt.plot(x, med_d[1, :], color='blue', linewidth=3, label=r'$7 \times \Delta W$')
        plt.plot([4, 4], [5, 23], color='black', linewidth=1)
        plt.fill_between(x, med_d[0, :], med_d[2, :], color='blue', alpha=0.5)
        plt.legend(fontsize=fontsize)
        plt.tight_layout()

        if simu == 'cs':
            plt.savefig('output/kmeans_optimal_k.png')
        elif simu == 'rand':
            plt.savefig('output/kmeans_optimal_k_rand.png')

        #plt.show()
        plt.cla()
        plt.clf()
        plt.close()

    # Plot the fraction of matching cells (total and average) vs. lambda
    if lambdath:

        # Read in 
        lambdas = pickle.load(open(lambdas_file, 'rb'))

        all_l1 = []
        all_l2 = []
        all_t1 = []
        all_t2 = []

        lth_max = 0.25

        for i in range(0, 10):
            x = lambdas[i][0,:]    
            t1 = lambdas[i][1,:]    
            t2 = lambdas[i][2,:]    
            
            i1 = t.find_max_index(t1)
            i2 = t.find_max_index(t2)

            # Filter the curves 
            if x[i1] < lth_max:

                print(x[i1], x[i2])
                all_l1.append(x[i1])
                all_l2.append(x[i2])
                all_t1.append(t1)
                all_t2.append(t2)

        n_l1 = len(all_l1)
        n_pts = len(x)
        all_t1 = np.array(all_t1)
        all_t2 = np.array(all_t2)
        med_l1 = np.zeros((3, n_pts))
        med_l2 = np.zeros((3, n_pts))

        for i in range(0, n_pts):
            med_l1[0, i] = np.min(all_t1[:, i])
            med_l1[1, i] = np.median(all_t1[:, i])
            med_l1[2, i] = np.max(all_t1[:, i])
            med_l2[0, i] = np.min(all_t2[:, i])
            med_l2[1, i] = np.median(all_t2[:, i])
            med_l2[2, i] = np.max(all_t2[:, i])
        
        str_l1 = '%.3f %.3f' % (np.median(all_l1), np.std(all_l1))
        str_l2 = '%.3f %.3f' % (np.median(all_l2), np.std(all_l2))
        print(str_l1)
        print(str_l2)

        plt.figure(figsize=(6,6))
        plt.grid(False)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xlabel(r'$\lambda _{thr}$', fontsize=fontsize)
        plt.ylabel(r'$f_m$', fontsize=fontsize)
        plt.plot(x, med_l1[1, :], color='black', linewidth=3, label='avg')
        plt.fill_between(x, med_l1[0, :], med_l1[2, :], color='grey', alpha=0.5)
        plt.plot(x, med_l2[1, :], color='blue', linewidth=3, label='tot')
        plt.fill_between(x, med_l2[0, :], med_l2[2, :], color='blue', alpha=0.5)
        plt.legend(fontsize=fontsize)
        plt.tight_layout()
        if simu == 'cs':
            plt.savefig('output/kmeans_matching_lambdas.png')
        elif simu == 'rand':
            plt.savefig('output/kmeans_matching_lambdas_rand.png')
        #plt.show()
        plt.cla()
        plt.clf()


if __name__ == "__main__":
    """ MAIN PROGRAM - compute k-means """

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

    halos_web(); exit()

    #file_base = '/home/edoardo/CLUES/DATA/Vweb/512/CSV/'
    file_base = '/home/edoardo/CLUES/DATA/VWeb/FullBox/'
    #file_base = '/home/edoardo/CLUES/TEST_DATA/VWeb/'
    #file_base = '/home/edoardo/CLUES/DATA/VWeb/'
    #file_base = '/home/edoardo/CLUES/DATA/LGF/512/05_14/'
    #file_base = '/home/edoardo/CLUES/DATA/FullBox/17_11/'
    #file_base = '/media/edoardo/data1/DATA/VWeb/512/full/'
    #web_file = 'vweb_00_10.000032.Vweb-csv'; str_grid = '_grid32'; grid = 32
    #web_file = 'vweb_00_10.000064.Vweb-csv'; str_grid = '_grid64'; grid = 64
    #web_file = 'vweb_00_10.000128.Vweb-csv'; str_grid = '_grid128'; grid = 128
    web_file = 'vweb_00.000128.Vweb-csv'; str_grid = '_grid128'; grid = 128
    #web_file = 'vweb_01.000128.Vweb-csv'; str_grid = '_grid128'; grid = 128
    #web_file = 'vweb_2048.000128.csv'; str_grid = '_grid128'; grid = 128
    #web_file = 'vweb_512_128_054.000128.Vweb.csv'; str_grid = '_grid128'; grid = 128
    #web_file = 'vweb_17_11.ascii.000128.Vweb-csv'; str_grid = '_grid128'; grid = 128
    #web_file = 'vweb_00_10.000032.Vweb-csv'; str_grid = '_grid32'; grid = 32
    #web_file = 'vweb_17_11_256.ascii.000256.Vweb-csv'; str_grid = '_grid256'; grid = 256
    #web_file = 'vweb_2048.000128.Vweb-ascii'; str_grid = '_grid128'; grid = 128
    #web_file = 'vweb_25_15.000128.Vweb-csv'; str_grid = '_grid128'; grid = 128
    #web_file = 'vweb_00_10.000128.Vweb-csv'; str_grid = '_grid128'; grid = 128;

    web_kmeans_file = file_base + 'vweb_kmeans.csv'
    #kmeans_file = 'output/kmeans.pkl'
    #kmeans_file = 'output/kmeans_new.pkl'
    kmeans_file = 'output/kmeans_k4.pkl'

    halo_webtype_file = 'output/halos_webtype.csv'

    #box = 500.0e+3; thick = 7.0e+3
    #box = 500.0; thick = 5.0
    box = 100.0; thick = 2.0

    # Use random or constrained simulation set
    #simu = 'cs'; n_steps = 10
    simu = 'rand'; n_steps = 5

    #centers(); exit()
    #alternative_web(); exit()
    #wt.elbow_visualize()
    #plot_cmp()
    #plot_halo_webtype(file_webtype=halo_webtype_file, correct_type=False); exit()
    #web_stats(simu=simu); exit()

    #ks, vs = multiple_web(file_name=file_base+web_file, lambdas=[0.18, 0.21], lambdath=False, elbow=False); 
    #print(vs)

    if simu == 'rand':
        file_scores = 'output/kmeans_rand_scores_fullweb.pkl'; file_lambdas = 'output/kmeans_rand_lambdas_fullweb.pkl'
        lambdas = []
    elif simu == 'cs':
        file_scores = 'output/kmeans_scores_fullweb.pkl'; file_lambdas = 'output/kmeans_lambdas_fullweb.pkl'
        lambdas = []

    centers = True
    lambdath = True
    elbow = False

    #multiple_web(bootstrap=False, scores_out=file_scores, lambdas_out=file_lambdas, lambdas=[0.21, 0.24], simu=simu, elbow=False, lambdath=False, centers=True); exit()
    #multiple_web(bootstrap=False, scores_out=file_scores, lambdas_out=file_lambdas, lambdas=[0.21], simu=simu, elbow=elbow, lambdath=lambdath, centers=centers, n_steps=n_steps); exit()
    #multiple_web(bootstrap=False, scores_out=file_scores, lambdas_out=file_lambdas, lambdas=[0.21, 0.24], elbow=True, lambdath=False); exit()
    #plot_multiple_cmp(elbow=True, scores_file=file_scores, lambdath=True, lambdas_file=file_lambdas); exit()
    #plot_multiple_cmp(elbow=True, scores_file=file_scores, lambdath=False, lambdas_file=file_lambdas); exit()
    
    web_df = pd.read_csv(file_base + web_file, dtype=float)
    #web_df = gen_coord(data=web_df)
    #web_df.to_csv(file_base + web_file)
    #print(len(web_df))
    #print(web_df.head())

    # Rescale the coordinates
    web_df['x'] = web_df['x'].values - box * 0.5
    web_df['y'] = web_df['y'].values - box * 0.5

    #threshold_list = [0.0, 0.1, 0.2]
    threshold_list = [0.22, 0.26]
    #threshold_list = [0.21, 0.24]
    #threshold_list = [0.18, 0.21]
    #threshold_list = []
    cols = []

    # Check out that the vweb coordinates should be in Mpc units
    if normalize == True:
        norm = 1.0e-3   # kpc to Mpc
        print('Norm: ', norm) 

        web_df['l1'] = web_df['l1'] / norm
        web_df['l2'] = web_df['l2'] / norm
        web_df['l3'] = web_df['l3'] / norm

    if plotStd == True: 
        for thresh in threshold_list:
            
            colth = str(thresh)
            cols.append(colth)
            print('std vweb for : ', colth)
            web_df['env'] = web_df[['l1', 'l2', 'l3']].apply(lambda x: wt.find_env(*x, thresh), axis=1)
            web_df[colth] = web_df[['l1', 'l2', 'l3']].apply(lambda x: wt.find_env(*x, thresh), axis=1)
            #print(web_df.head())
            #vweb_env = np.array(web_df['env'].values, dtype=int)

            f_out = 'output/kmeans_rand_smooth' + str_grid + '_l' + str(thresh)
            #f_out = 'output/kmeans_oldvweb' + str_grid + '_l' + str(thresh)
            title_str = r'$\lambda _{thr}=$' + str(thresh)
            web_df_slice = wt.plot_vweb_smooth(fout=f_out, data=web_df, thresh=thresh, grid=grid, box=box, thick=thick, do_plot=True, title=title_str, use_thresh=True)

            '''
            #web_df_slice = wt.plot_vweb(fout=f_out, data=web_df, thresh=thresh, grid=grid, box=box, thick=thick, do_plot=True, title=title_str, use_thresh=True)
            #web_df_slice = wt.plot_vweb(fout=f_out, data=web_df_loc, thresh=thresh, grid=grid, box=box, thick=thick, do_plot=False, title=title_str)
            '''
            f_out = 'output/kmeans_' + simu + '_' + str_grid + '_l' + str(thresh)
            wt.plot_lambda_distribution(data=web_df, base_out=f_out, x_axis=True)

    cols_select = ['l1', 'l2', 'l3']; vers = ''; str_kmeans = r'$k$-means $\lambda$s'

    n_clusters = 4
    n_init = 101

    if simu == 'cs':
        kmeans_file = 'output/kmeans_k' + str(n_clusters) + '.pkl'
        all_avg_file = 'output/kmeans_all_avg.pkl'; all_tot_file = 'output/kmeans_all_tot.pkl'
    elif simu == 'rand':
        kmeans_file = 'output/kmeans_rand_k' + str(n_clusters) + '.pkl'
        all_avg_file = 'output/kmeans_all_rand_avg.pkl'; all_tot_file = 'output/kmeans_all_rand_tot.pkl'

    if read_kmeans:
        print('Loading k-means from ', kmeans_file)
        kmeans = pickle.load(open(kmeans_file, 'rb'))
        centers = kmeans.cluster_centers_
        web_df['env'] = kmeans.labels_
        print('Done.')

    else:
        print('Running kmeans...')
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
        kmeans.fit(web_df[cols_select])
        centers = kmeans.cluster_centers_
        pickle.dump(kmeans, open(kmeans_file, 'wb'))
        web_df['env'] = kmeans.labels_
        web_df.to_csv(web_kmeans_file)
        print('Saving kmeans to ', kmeans_file)
    
    #print('Assigning halos to nearest node / environment type')
    #halos_web = wt.assign_halos_to_environment_type(vweb=vweb_env, kmeans=kmeans_env, snap_end=1)
    #halos_web.to_csv(halo_webtype_file)
    #print('Done')
    #wt.evaluate_metrics(data=web_df[['l1', 'l2', 'l3']], elbow=True)

    envs = [i for i in range(0, n_clusters)]
    f_out = 'output/kmeans_' + simu + '_' + str_grid + '_k' + str(n_clusters) + '.png'
    envirs_sort, colors_sort, number_sort = order_by_delta(n_clusters=n_clusters, web_df=web_df, centers=centers)

    wt.plot_lambda_distribution(data=web_df, base_out=f_out, x_axis=True)
    title_str = r'$k$-means, k='+str(n_clusters)
    wt.plot_vweb_smooth(data=web_df, fout=f_out, grid=128, use_thresh=False, ordered_envs=number_sort, title=title_str, envs=envs)
    
    web_df['envk_std'] = kmeans.labels_
    web_df = wt.order_kmeans(data=web_df, nk=n_clusters)
    f_out = 'output/evs_project_' + simu + '_'
    wt.plot2d(data=web_df, f_out=f_out); exit()

    thresh = [i * 0.01 for i in range(0, 60)]
    all_avg = []
    all_tot = []
    all_avg_m = []
    all_tot_m = []

    for th in thresh:
        web_df = wt.std_vweb(data=web_df, thresh=th)
        avg, tot, avg_m, tot_m = wt.compare_vweb_kmeans(vweb=web_df, l=th)
        all_avg.append(avg)
        all_tot.append(tot)
        all_avg_m.append(avg_m)
        all_tot_m.append(tot_m)
   
    print(web_df.head())
    web_df['envk_std'] = kmeans.labels_
    web_df = wt.order_kmeans(data=web_df)

    cols.append('envk')
    #print(cols)
    #print(web_df.head())

    print('Plotting densities...')
    f_out = 'output/web_dens_' + simu + '_'
    wt.plot_densities(data=web_df, cols=cols, f_out=f_out)

'''
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
