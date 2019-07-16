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
    epoch = 10000
    model.test(dataset, '%s_%d' % (parameters.model_file_location, epoch))
    model.plot_test_result(dataset, 'fig_%d' % epoch).show()
