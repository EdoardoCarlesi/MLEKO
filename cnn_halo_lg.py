'''
    CluesML
    Machine Learning tools for the CLUES project

    (C) Edoardo Carlesi 2020
    https://github.com/EdoardoCarlesi/CluesML
'''

import tools as t
import pandas as pd
import numpy as np
import keras as ks

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import image

# Choose wether to load or save the neural network model
train_model = False
local_path = '/home/edoardo/CLUES/CluesML/'

# Dataset path
test_path = '/home/edoardo/CLUES/PyRCODIO/output/100x100/test/'
train_path = '/home/edoardo/CLUES/PyRCODIO/output/100x100/train/'

# Initialize the CNN
classifier = Sequential()

# Add the first convolutional layer
# n_feat ----> number of feature detectors, n_dim x n_dim dimension. 
n_feat = 16
n_feat_dim = 4 

# Assuming input is n_input x n_input, the n_input_dim is set to 1 (b/w image)
n_input = 100
n_input_dim = 1

classifier.add(Convolution2D(n_feat, n_feat_dim, n_feat_dim, input_shape=(n_input, n_input, n_input_dim), activation='relu'))

# Pooling step: reduce the size of the feature maps, specify the size of the subtable as n_pool_dim
n_pool_dim = 2

classifier.add(MaxPooling2D(pool_size = (n_pool_dim, n_pool_dim)))

# Second convolutional layer
classifier.add(Convolution2D(n_input, n_input_dim, n_input_dim, activation='relu'))
classifier.add(MaxPooling2D(pool_size = (n_pool_dim, n_pool_dim)))

# Flattening: take all the pooled feature map and put it into a vector
classifier.add(Flatten())

# Build the fully connected ANN. First hidden layer
n_units = 32
classifier.add(Dense(units=n_units, activation='relu'))

# Output layer
classifier.add(Dense(units=1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Increase the data size
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.3,
                                   zoom_range = 0.3)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Set some properties of the training
batch_size = 8
n_epocs = 5
steps_per_epoch = 100
validation_steps = 50

# Save (or load) the CNN classifier
model_file_name = local_path + '/models/halo_lg_classifier_pix' + str(n_input) + '.keras'
#model_file_name = local_path + '/models/halo_lg_classifier_v1_pix' + str(n_input) + '.keras'

if train_model == True:
    
    # Set grayscale for the color mode to avoid the three RGB layers
    training_set = train_datagen.flow_from_directory(train_path, 
                                                 target_size = (n_input, n_input),
                                                 batch_size = batch_size,
                                                 color_mode = 'grayscale',
                                                 class_mode = 'binary')

    test_set = test_datagen.flow_from_directory(test_path, 
                                            target_size = (n_input, n_input),
                                            batch_size = batch_size,
                                            color_mode = 'grayscale',
                                            class_mode = 'binary')

    classifier.fit_generator(training_set,
                         steps_per_epoch = steps_per_epoch,
                         epochs = n_epocs,
                         validation_data = test_set,
                         validation_steps = validation_steps)

    classifier.save(model_file_name)

else:
#    img_path = []
#    img_path.append('/home/edoardo/CLUES/PyRCODIO/output/cluster_62_14_09.12.rho_no_labels_SGYSGZ.png')
#    img_path.append('/home/edoardo/CLUES/PyRCODIO/output/lg_34_13_09_rho_no_labels_SGZSGX.png')

    path_lg='/home/edoardo/CLUES/PyRCODIO/output/test_set/lg/'
    path_cl='/home/edoardo/CLUES/PyRCODIO/output/test_set/cluster/'
    img_path_lg = t.find_images_in_folder(path=path_lg)
    img_path_cl = t.find_images_in_folder(path=path_cl)
    #print(img_path_lg)

    #print(model_file_name)
    results_lg = t.check_cluster_lg_classifier(model_file_name=model_file_name, imgs=img_path_lg, n_input=n_input) #, color_mode='grayscale')   
    results_cl = t.check_cluster_lg_classifier(model_file_name=model_file_name, imgs=img_path_cl, n_input=n_input) #, color_mode='grayscale')   
    
    n_lg = len(img_path_lg)
    n_cl = len(img_path_cl)

    n_res_lg = np.sum(results_lg)    
    n_res_cl = np.sum(results_cl)

    print('N LG: ', n_lg, ' result: ', n_res_lg, ' accuracy: ', float(n_lg - n_res_lg)/n_lg)
    print('N Cluster: ', n_cl, ' result: ', n_res_cl, ' accuracy: ', float(n_res_cl)/n_cl)








