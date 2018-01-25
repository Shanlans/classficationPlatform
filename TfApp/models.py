# -*- coding: utf-8 -*-
# Created By Shanlan on 22/1/2018

import tensorflow as tf
import baseLayer


class models(object):
    
    def __init__(self,
                 x,
                 filterSizes,
                 outputChannels,
                 pooling,
                 padding='SAME',
                 bn=False,
                 activateFn=tf.nn.relu,
                 bValue = 0.0):
        '''
        Shortcut for creating a 2D Convolutional Neural Network in one line
        
        Stacks multiple conv2d layers, with arguments for each layer defined in a list.        
        If an argument is left as None, then the conv2d defaults are kept
        :param filterSizes: int. assumes square filter
        :param outputChannels: int
        :param stride: int
        :param padding: 'VALID' or 'SAME'
        :param activationFn: tf.nn function
        :param bValue: float
        :param sValue: float
        '''  
        
        
        self.x = x
        self.filterSize = filterSize
        self.outputChannel = outputChannel
        self.pooling = pooling
        self.padding = padding
        self.bn = bn
        self.activateFn = activateFn
        self.bValue = bValue
        
        self.depth = len(filterSize)
        
        self.network = self._Network(x)
    
    
    def _Network(self,
                x):
        layers = baseLayer.Layers(x)
        
        def convnet(self, filterSize, outputChannels, stride=None, padding=None, activationFn=None, bValue=None, sValue=None, bn=None, trainable=True):
              
        # Number of layers to stack
        depth = len(filter_size)
        
        # Default arguments where None was passed in
        if stride is None:
            stride = np.ones(depth)
        if padding is None:
            padding = ['SAME'] * depth
        if activation_fn is None:
            activation_fn = [tf.nn.relu] * depth
        if b_value is None: 
            b_value = np.zeros(depth)
        if s_value is None:
            s_value = np.ones(depth)
        if bn is None:
            bn = [True] * depth 
            
        # Make sure that number of layers is consistent
        assert len(output_channels) == depth
        assert len(stride) == depth
        assert len(padding) == depth
        assert len(activation_fn) == depth
        assert len(b_value) == depth
        assert len(s_value) == depth
        assert len(bn) == depth
        
        # Stack convolutional layers
        for l in range(depth):
            self.conv2d(filter_size=filter_size[l],
                        output_channels=output_channels[l],
                        stride=stride[l],
                        padding=padding[l],
                        activation_fn=activation_fn[l],
                        b_value=b_value[l], 
                        s_value=s_value[l], 
                        bn=bn[l], trainable=trainable)
