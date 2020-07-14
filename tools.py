'''
    CluesML
    Machine Learning tools for the CLUES project

    (C) Edoardo Carlesi 2020
    https://github.com/EdoardoCarlesi/CluesML
'''

from keras.models import Sequential
from keras.preprocessing.image import image
import pandas as pd
import numpy as np
import keras as ks
import glob, os
import cv2


'''
    Check the shape of an input image
'''
def check_image(img):
    img_test = '/home/edoardo/CLUES/PyRCODIO/output/cluster_62_14_09.12.rho_no_labels_SGYSGZ.png'
    f = cv2.imread(img_test)
    print(img,  ' has shape: ', f.shape)

    return f.shape

'''
    Load a previously trained cluster/lg classifier and use it to classify a new image or list of images
'''
def check_cluster_lg_classifier(model_file_name=None, imgs=[], n_input=100, color_mode='grayscale', verbose=False):
    print('Loading Keras model: ', model_file_name)
    classifier = ks.models.load_model(model_file_name)
    class_result = []

    if isinstance(imgs, list) == False:
        imgs = [imgs]

    for img_path in imgs:
        img_test = image.load_img(img_path, target_size = (n_input, n_input), color_mode = color_mode)
        img_test = image.img_to_array(img_test)
        img_test = np.expand_dims(img_test, axis=0)
        result = classifier.predict(img_test)
    
        if result[0][0] == 0.0:
            img_class = 'Cluster'
            img_result = 0
        else:
            img_class = 'LocalGroup'
            img_result = 1

        if verbose == True:
            print('Image: ', img_path, ' is a ', img_class, ' (', img_result, ')')

        class_result.append(img_result)

    return class_result

'''
    Spit out the path to all the files within a given folder
'''
def find_images_in_folder(path=None, extension='png'):
    os.chdir(path)
    img_list = []

    for img in glob.glob('*.'+extension):
        img_list.append(path + img)

    return img_list




