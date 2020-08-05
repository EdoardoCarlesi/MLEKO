'''
    CluesML
    Machine Learning tools for the CLUES project

    (C) Edoardo Carlesi 2020
    https://github.com/EdoardoCarlesi/CluesML
'''

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
import glob, os
import cv2

'''
    Do a PCA analysis of a dataset
'''
def data_pca(data=None, columns=None):

    print('Doing PCA reduction of dataset: ', columns)

    # Initialize PCA
    n_components = len(columns)
    pca = PCA(n_components = n_components)

    # Select these data from the full dataframe
    X = data[columns]

    # Normalize the data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Transform
    principal_components = pca.fit_transform(X)

    # Put the PCA transformed data into a DataFrame
    pc_df = pd.DataFrame(data=principal_components, columns=columns)

    # Plot some information
    print('PCA explained variance ratio is: ', pca.explained_variance_ratio_)

    return pc_df

'''
    Check the shape of an input image
'''
def check_image(img):
    f = cv2.imread(img)
    print(img,  ' has shape: ', f.shape)

    return f.shape

'''
    Spit out the path to all the files within a given folder
'''
def find_images_in_folder(path=None, extension='png'):
    os.chdir(path)
    img_list = []

    for img in glob.glob('*.'+extension):
        img_list.append(path + img)

    return img_list

'''
    Mean absolute percentage error
'''
def MAPE(y_true, y_pred):
    d_y = np.abs(y_true - y_pred)
    d_y = d_y / y_true

    return d_y


