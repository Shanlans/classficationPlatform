# -*- coding: utf-8 -*-
# Created By Shanlan on 22/1/2018

import tensorflow as tf
import numpy as np
import baseLayer


class models(object):
    
    def __init__(self,
                 x,
                 filterSizes,
                 outputChannels,
                 trainable,
                 padding=None,
                 bn=None,
                 activationFn=None,
                 bValue=None
                 ):
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
        :param trainable: train or validation
        '''  
        
        
        self.x = x
        print(x.get_shape())
        self.__filterSizes = filterSizes
        self.__outputChannels = outputChannels
        self.__padding = padding
        self.__bn = bn
        self.__activationFn = activationFn
        self.__bValue = bValue
        self.__trainable = trainable
        
        self.depth = len(filterSizes)
        
        self.network = self._Network(x)
    
    
    def _Network(self,x):
        self.layers = baseLayer.Layers(x)
        filterSizes = self.__filterSizes
        outputChannels = self.__outputChannels
        padding = self.__padding
        bn = self.__bn
        activationFn = self.__activationFn
        bValue = self.__bValue
        trainable = self.__trainable
        
              
        # Number of layers to stack
        depth = self.depth
        stride = None
        sValue = None
        # Default arguments where None was passed in
        if stride is None:
            stride = np.ones(depth)
        if padding is None:
            padding = ['SAME'] * depth
        if activationFn is None:
            activationFn = [tf.nn.relu] * depth
        if bValue is None: 
            bValue = np.zeros(depth)
        if sValue is None:
            sValue = np.ones(depth)
        if bn is None:
            bn = [False] * depth 
            
        # Make sure that number of layers is consistent
        assert len(outputChannels) == depth
        assert len(stride) == depth
        assert len(padding) == depth
        assert len(activationFn) == depth
        assert len(bValue) == depth
        assert len(sValue) == depth
        assert len(bn) == depth
        
        # Stack convolutional layers
        for l in range(depth):
            self.layers.Conv2D(filterSize=filterSizes[l],
                          outputChannel=outputChannels[l],
                          stride=stride[l],
                          padding=padding[l],
                          activationFn=activationFn[l],
                          bValue=bValue[l], 
                          sValue=sValue[l], 
                          bn=bn[l], 
                          trainable=trainable)
            if l == depth-1:
                self.layers.AvgPool(globe=True)
            else:
                self.layers.MaxPool()
                
    def get_output(self):
        
        predicts = tf.nn.softmax(self.layers.get_output())
        return predicts