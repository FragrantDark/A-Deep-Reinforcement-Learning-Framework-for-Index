# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 09:53:23 2019

@author: qizhang
"""

import read_data
import train_model

from set_parameters import parameters

if __name__ == '__main__':

    dataset = read_data.Dataset(parameters)
    model = train_model.NNAgent(parameters)
    epoch = 140000
    # dataset.test_matrix_w = dataset.train_matrix_w[:dataset.n_test, :]
    # dataset.test_dataset = dataset.train_dataset[:dataset.n_test, :, :]
    model.test(dataset, '%s_%d' % (parameters.model_file_location, epoch))
    model.plot_test_result(dataset, 'fig_%d' % epoch).show()
