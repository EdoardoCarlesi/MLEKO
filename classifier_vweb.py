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
web_file = 'vweb_00_10.000032.Vweb-csv'
#web_file = 'vweb_00_10.000064.Vweb-csv'
#web_file = 'vweb_25_15.000064.Vweb-csv'
#web_file = 'vweb_01_10.000064.Vweb-csv'
#web_file = 'vweb_00_10.000128.Vweb-csv'

web_df = pd.read_csv(file_base + web_file)
#print(web_df.head())

cols_select = ['l1', 'l2', 'l3']

web_ev_df = web_df[cols_select]

kmeans = KMeans(n_clusters = 4, n_init = 10)
kmeans.fit(web_ev_df)

print(kmeans.cluster_centers_)
print(type(kmeans.labels_))
web_df['env'] = kmeans.labels_
ntot = len(web_df)

for i in range(0, 4):
    n = len(web_df[web_df['env'] == i]) 
    print(n, ' perc: ', n/ntot)

'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(web_ev_df['l1'], web_ev_df['l2'], web_ev_df['l3'], c = kmeans.labels_)

plt.show()
'''

