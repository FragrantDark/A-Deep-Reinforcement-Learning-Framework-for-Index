# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 14:21:31 2019

@author: dlymhth
"""


class Parameters:
    def __init__(self):
        # Input data
        self.varieties = ['a', 'i', 'j', 'jm',
                          'm', 'p', 'y']
        self.features = ['new.price',
                         'last.minute.high.price',
                         'last.minute.low.price']
        self.train_data_rate = 0.7

        # Model structure
        self.n_timesteps = 50
        self.n_batch = 32

        self.height_cov1 = 3
        self.n_cov1_core = 2
        self.height_cov2 = self.n_timesteps - self.height_cov1 + 1
        self.n_cov2_core = 20
        self.height_cov3 = self.n_cov2_core + 1

        # Training
        self.n_epochs = int(1e6)
        self.display_step = int(1e3)
        self.checkpoint = int(1e4)

        self.start_learning_rate = 0.001
        self.decay_steps = 10e3
        self.decay_rate = 0.1

        self.commission_rate = 0.0001

        # File location
        self.root_dir = '/Users/zhangqi/PycharmProjects/A-Deep-Reinforcement-Learning-Framework-for-Index'
        self.data_file_location = self.root_dir + '/data/'
        self.model_file_location = self.root_dir + '/model/model_c%f_e%d' % (self.commission_rate, self.n_epochs)
        self.figure_file_location = self.root_dir + '/figures/'
        self.logfile = self.root_dir + '/log/t.log'


parameters = Parameters()
