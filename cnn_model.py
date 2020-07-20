'''
    CluesML
    Machine Learning tools for the CLUES project

    (C) Edoardo Carlesi 2020
    https://github.com/EdoardoCarlesi/CluesML
'''



class CNN_Model():

    n_feat = 16
    n_feat_dim = 4 
    n_pool_dim = 3 
    n_units = 32
    n_input = 100
    n_input_dim = 1
    batch_size = 8
    n_epochs = 5
    steps_per_epoch = 100
    validation_steps = 50
    shear_range = 0.2
    zoom_range = 0.2
    test_path = '/home/edoardo/CLUES/PyRCODIO/output/100x100/test/'
    train_path = '/home/edoardo/CLUES/PyRCODIO/output/100x100/train/'

    def __init__(self, version=0):

        if version == '0':
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





