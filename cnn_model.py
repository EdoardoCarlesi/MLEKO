"""
    MLEKO
    Machine Learning Environment for KOsmology

    (C) Edoardo Carlesi 2020
    https://github.com/EdoardoCarlesi/MLEKO
"""




class CNN_Model():

    # Number of feature detectors
    n_feat = 16

    # Size of the matrix for the feature detector (square)
    n_feat_dim = 4 

    # Size of the pooling layer matrix (square)
    n_pool_dim = 3 

    # Units in the first ANN layer
    n_units = 32

    # Size of th input image (rescale to this value)
    n_input = 100

    # Dimension of the input (1 = b/3, 3 = rgb)
    n_input_dim = 1

    # Size of the training batch
    batch_size = 8

    # How many training epochs
    n_epochs = 5

    # Steps of training per epoch
    steps_per_epoch = 100

    # Validation steps
    validation_steps = 50

    # Shear rotation goes between - and + this value
    shear_range = 0.2

    # Zoom range
    zoom_range = 0.2

    # Datasets
    test_path = '/home/edoardo/CLUES/PyRCODIO/output/100x100/test/'
    train_path = '/home/edoardo/CLUES/PyRCODIO/output/100x100/train/'

    def __init__(self, version=0):

        if version == '0':

            '''
                N LG:  312  result:  186  accuracy:  0.40384615384615385
                N Cluster:  524  result:  2  accuracy:  0.9961832061068703
            '''

            self.n_feat = 16
            self.n_feat_dim = 4 
            self.n_pool_dim = 3 
            self.n_units = 32
            self.n_input = 100
            self.n_input_dim = 1
            self.batch_size = 8
            self.n_epochs = 5
            self.steps_per_epoch = 100
            self.validation_steps = 50
            self.shear_range = 0.2
            self.zoom_range = 0.2
            self.test_path = '/home/edoardo/CLUES/PyRCODIO/output/100x100/test/'
            self.train_path = '/home/edoardo/CLUES/PyRCODIO/output/100x100/train/'

        elif version == '1':

            '''
                N LG:  312  result:  22  accuracy:  0.9294871794871795
                N Cluster:  524  result:  0  accuracy:  1.0
            '''

            self.n_feat = 24
            self.n_feat_dim = 3 
            self.n_pool_dim = 2 
            self.n_units = 24
            self.n_input = 64
            self.n_input_dim = 1 
            self.batch_size = 10
            self.n_epochs = 6
            self.steps_per_epoch = 120
            self.validation_steps = 40
            self.shear_range = 0.2
            self.zoom_range = 0.2
            self.test_path = '/home/edoardo/CLUES/PyRCODIO/output/120x120/test/'
            self.train_path = '/home/edoardo/CLUES/PyRCODIO/output/120x120/train/'

        elif version == '2':

            '''
                N LG:  312  result:  39  accuracy:  0.875
                N Cluster:  524  result:  0  accuracy:  1.0
            '''

            self.n_feat = 8
            self.n_feat_dim = 3 
            self.n_pool_dim = 2 
            self.n_units = 16
            self.n_input = 128
            self.n_input_dim = 1
            self.batch_size = 8
            self.n_epochs = 4
            self.steps_per_epoch = 80
            self.validation_steps = 20
            self.shear_range = 0.5
            self.zoom_range = 0.5
            self.test_path = '/home/edoardo/CLUES/PyRCODIO/output/120x120/test/'
            self.train_path = '/home/edoardo/CLUES/PyRCODIO/output/120x120/train/'

        elif version == '3':

            '''
                N LG:  312  result:  193  accuracy:  0.3814102564102564
                N Cluster:  524  result:  1  accuracy:  0.9980916030534351
            '''

            self.n_feat = 20
            self.n_feat_dim = 2 
            self.n_pool_dim = 2 
            self.n_units = 32
            self.n_input = 80
            self.n_input_dim = 1
            self.batch_size = 10
            self.n_epochs = 4
            self.steps_per_epoch = 80
            self.validation_steps = 20
            self.shear_range = 0.5
            self.zoom_range = 0.5
            self.test_path = '/home/edoardo/CLUES/PyRCODIO/output/120x120/test/'
            self.train_path = '/home/edoardo/CLUES/PyRCODIO/output/120x120/train/'

        elif version == '4':

            '''
                N LG:  312  result:  29  accuracy:  0.907051282051282
                N Cluster:  524  result:  0  accuracy:  1.0
            '''

            self.n_feat = 32
            self.n_feat_dim = 3 
            self.n_pool_dim = 2 
            self.n_units = 16
            self.n_input = 64
            self.n_input_dim = 1
            self.batch_size = 10
            self.n_epochs = 5
            self.steps_per_epoch = 100
            self.validation_steps = 25
            self.shear_range = 0.2
            self.zoom_range = 0.2
            self.test_path = '/home/edoardo/CLUES/PyRCODIO/output/120x120/test/'
            self.train_path = '/home/edoardo/CLUES/PyRCODIO/output/120x120/train/'

        elif version == '5':

            '''
                N LG:  312  result:  193  accuracy:  0.3814102564102564
                N Cluster:  524  result:  1  accuracy:  0.9980916030534351
            '''

            self.n_feat = 20
            self.n_feat_dim = 2 
            self.n_pool_dim = 2 
            self.n_units = 32
            self.n_input = 80
            self.n_input_dim = 1
            self.batch_size = 10
            self.n_epochs = 5
            self.steps_per_epoch = 200
            self.validation_steps = 50
            self.shear_range = 0.25
            self.zoom_range = 0.25
            self.test_path = '/home/edoardo/CLUES/PyRCODIO/output/120x120/test/'
            self.train_path = '/home/edoardo/CLUES/PyRCODIO/output/120x120/train/'


