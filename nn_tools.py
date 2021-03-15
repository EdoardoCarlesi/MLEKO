"""
    MLEKO
    Machine Learning Environment for KOsmology

    (C) Edoardo Carlesi 2020
    https://github.com/EdoardoCarlesi/MLEKO
"""


import keras as ks
import pandas as pd
import numpy as np
import glob, os
import cv2

'''
    Load a previously trained cluster/lg classifier and use it to classify a new image or list of images
'''
def check_cluster_lg_classifier(model_file_name=None, imgs=[], n_input=100, color_mode='grayscale', verbose=False, class_mode=None):
    print('Loading Keras model: ', model_file_name)
    classifier = ks.models.load_model(model_file_name)
    class_result = []

    if isinstance(imgs, list) == False:
        imgs = [imgs]

    for img_path in imgs:
        img_test = image.load_img(img_path, target_size = (n_input, n_input), color_mode = color_mode)
        img_test = image.img_to_array(img_test)/255.
        #print(img_test)
        #image.save_img('/home/edoardo/test_di_merda255.jpg', img_test)

        img_test = np.expand_dims(img_test, axis=0)
        result = classifier.predict(img_test)

        if verbose == True:
            print(class_mode, ' classifier result: ', result)

        if class_mode == 'binary':

            if result[0][0] < 0.5:
                img_class = 'Cluster'
                img_result = 0
            else:
                img_class = 'LocalGroup'
                img_result = 1

        if class_mode == 'categorical':

            if result[0][0] < 0.5:
                img_class = 'Cluster'
                img_result = 0
            else:
                img_class = 'LocalGroup'
                img_result = 1

        if verbose == True:
            print('Image: ', img_path, ' is a ', img_class, ' (', img_result, ')')

        class_result.append(img_result)

    return class_result


