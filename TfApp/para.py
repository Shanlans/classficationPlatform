# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:03:55 2018

@author: yedexian
"""
import os

trainSwitch = True
babysitting = True
max_steps = 100
learning_rate = 0.01
mini_batch_size = 16
data_dir = os.path.join(os.getenv('DATADIR', 'Database'),'IR_data')