"""
    MLEKO
    Machine Learning Ecosystem for KOsmology

    (C) Edoardo Carlesi 2020
    https://github.com/EdoardoCarlesi/MLEKO
"""


import pandas as pd
import tools as t



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



