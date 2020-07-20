'''
    CluesML
    Machine Learning tools for the CLUES project

    (C) Edoardo Carlesi 2020
    https://github.com/EdoardoCarlesi/CluesML
'''

from cnn_model import CNN_Model
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
#train_model = True

class_mode = 'binary'
#class_mode = 'categorical'

local_path = '/home/edoardo/CLUES/CluesML/'

"""
    CNN architecture versions:
    - v0: n_feat_dim = 4, n_feat = 16, n_input = 100, n_pool_dim = 3, n_units = 32. Conv2d, pooling, conv2d, pooling, Dense, Output
        Training: batch_size = 8, n_epochs = 5, steps_per_epoch = 100, validation_steps = 50; shear_range = 0.3; zoom_range = 0.3
        test_path = '/home/edoardo/CLUES/PyRCODIO/output/100x100/test/'
        train_path = '/home/edoardo/CLUES/PyRCODIO/output/100x100/train/'

    - v1: n_feat_dim = 3, n_feat = 24, n_input = 64, n_pool_dim = 2, n_units = 24. Conv2d, pooling, conv2d, pooling, Dense, Output
        Training: batch_size = 10, n_epochs = 6, steps_per_epoch = 120, validation_steps = 40; shear_range = 0.2; zoom_range = 0.2
        test_path = '/home/edoardo/CLUES/PyRCODIO/output/100x100/test/'
        train_path = '/home/edoardo/CLUES/PyRCODIO/output/100x100/train/'

    - v2: n_feat_dim = 4, n_feat = 24, n_input = 64, n_pool_dim = 2, n_units = 24. Conv2d, pooling, conv2d, pooling, Dense, Output
        Training: batch_size = 10, n_epochs = 6, steps_per_epoch = 120, validation_steps = 40; shear_range = 0.2; zoom_range = 0.2
        test_path = '/home/edoardo/CLUES/PyRCODIO/output/100x100/test/'
        train_path = '/home/edoardo/CLUES/PyRCODIO/output/100x100/train/'

    ... and so on. 
    Check the cnn_model.py source for all the details

"""

# Select different CNN versions
version = '5'

cm = CNN_Model(version = version)

# Initialize the CNN
classifier = Sequential()

# Add the first convolutional layer with input to n_input x n_input, the n_input_dim is set to 1 (b/w image)
classifier.add(Convolution2D(cm.n_feat, cm.n_feat_dim, cm.n_feat_dim, input_shape=(cm.n_input, cm.n_input, cm.n_input_dim), activation='relu'))

# Pooling step: reduce the size of the feature maps, specify the size of the subtable as n_pool_dim
classifier.add(MaxPooling2D(pool_size = (cm.n_pool_dim, cm.n_pool_dim)))

# Second convolutional layer
classifier.add(Convolution2D(cm.n_input, cm.n_input_dim, cm.n_input_dim, activation='relu'))
classifier.add(MaxPooling2D(pool_size = (cm.n_pool_dim, cm.n_pool_dim)))

# Flattening: take all the pooled feature map and put it into a vector
classifier.add(Flatten())

# Build the fully connected ANN. First hidden layer
classifier.add(Dense(units=cm.n_units, activation='relu'))

# Output layer
if class_mode == 'binary':
    classifier.add(Dense(units=1, activation='sigmoid'))

elif class_mode == 'categorical':
    classifier.add(Dense(units=2, activation='softmax'))
else:
    print('class_mode = ', class_mode, ' is not supported.')
    exit()

classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Increase the data size
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = cm.shear_range,
                                   zoom_range = cm.zoom_range)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Save (or load) the CNN classifier
if class_mode == 'binary':
    model_file_name = local_path + '/models/halo_lg_classifier_v' + version + '.keras'

elif class_mode == 'categorical':
    model_file_name = local_path + '/models/halo_lg_classifier_categorical_v' + version + '.keras'

if train_model == True:
    
    # Set grayscale for the color mode to avoid the three RGB layers
    training_set = train_datagen.flow_from_directory(cm.train_path, 
                                                 target_size = (cm.n_input, cm.n_input),
                                                 batch_size = cm.batch_size,
                                                 color_mode = 'grayscale',
                                                 class_mode = class_mode)

    test_set = test_datagen.flow_from_directory(cm.test_path, 
                                            target_size = (cm.n_input, cm.n_input),
                                            batch_size = cm.batch_size,
                                            color_mode = 'grayscale',
                                            class_mode = class_mode)

    classifier.fit_generator(training_set,
                         steps_per_epoch = cm.steps_per_epoch,
                         epochs = cm.n_epochs,
                         validation_data = test_set,
                         validation_steps = cm.validation_steps)

    print('Saving trained CNN to: ', model_file_name)
    classifier.save(model_file_name)

else:   # We load a pre-compiled, fitted and trained model
    
    # Set some properties of the test images
    verbose = False
    path_lg='/home/edoardo/CLUES/PyRCODIO/output/test_set/lg/'
    path_cl='/home/edoardo/CLUES/PyRCODIO/output/test_set/cluster/'

    # Find all the images within a given folder (images NOT used for the training)
    img_path_lg = t.find_images_in_folder(path=path_lg)
    img_path_cl = t.find_images_in_folder(path=path_cl)

    '''
    img_path_shit = ['/home/edoardo/Pictures/images.jpeg']
    results_lg = t.check_cluster_lg_classifier(model_file_name=model_file_name, imgs=img_path_shit, n_input=n_input, verbose=verbose, class_mode=class_mode) 

    #path_mix = '/home/edoardo/CLUES/PyRCODIO/output/test_set/mix/'
    path_mix = '/home/edoardo/CLUES/PyRCODIO/output/test_set/mix/'
    img_path_mix = t.find_images_in_folder(path=path_mix)
    results_mix = t.check_cluster_lg_classifier(model_file_name=model_file_name, imgs=img_path_mix, n_input=n_input, verbose=verbose, class_mode=class_mode) 
    '''

    # Now feed the images to the classifier and check the results
    results_lg = t.check_cluster_lg_classifier(model_file_name=model_file_name, imgs=img_path_lg[:], n_input=cm.n_input, verbose=verbose, class_mode=class_mode) 
    results_cl = t.check_cluster_lg_classifier(model_file_name=model_file_name, imgs=img_path_cl[:], n_input=cm.n_input, verbose=verbose, class_mode=class_mode) 
    
    n_lg = len(img_path_lg)
    n_cl = len(img_path_cl)

    n_res_lg = np.sum(results_lg)    
    n_res_cl = np.sum(results_cl)

    # Some final statistics
    print('N LG: ', n_lg, ' result: ', n_res_lg, ' accuracy: ', float(n_lg - n_res_lg)/n_lg)
    print('N Cluster: ', n_cl, ' result: ', n_res_cl, ' accuracy: ', 1.0 - float(n_res_cl)/n_cl)






