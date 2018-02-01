# -*- coding: utf-8 -*-
# created by Shanlan on 24/1/2018
import tensorflow as tf
#import tensorflow.contrib.layers as init
import logging
import numpy as np


class Layers:
    """
    A Class to facilitate network creation in TensorFlow.
    Methods: conv2d, deconv2d, cflatten, maxpool, avgpool, res_layer, noisy_and, batch_norm
    """
    def __init__(self, x):
        """
        Initialize model Layers.
        .input = numpy array
        .count = dictionary to keep count of number of certain types of layers for naming purposes
        """
        self.input = x  # initialize input tensor
        self.count = {'conv': 0, 'deconv': 0, 'fc': 0, 'flat': 0, 'mp': 0, 'up': 0, 'ap': 0, 'rn': 0}

    def Conv2D(self, filterSize, outputChannel, stride=1, padding='SAME', bn=True, activationFn=tf.nn.relu, bValue=0.0, sValue=1.0, trainable=True):
        """
        2D Convolutional Layer.
        :param filterSize: int. assumes square filter
        :param outputChannels: int
        :param stride: int
        :param padding: 'VALID' or 'SAME'
        :param activationFn: tf.nn function
        :param bValue: float
        :param sValue: float
        """
        self.count['conv'] += 1
        scope = 'conv_' + str(self.count['conv'])
        with tf.variable_scope(scope):
            # Conv function
            inputChannels = self.input.get_shape()[3]
            if filterSize == 0:  # outputs a 1x1 feature map; used for FCN
                filterSize = self.input.get_shape()[2]
                padding = 'VALID'
            outputShape = [filterSize, filterSize, inputChannels, outputChannel]
            w = self.WeightVariable(name='weights', shape=outputShape, trainable=trainable)
            self.input = tf.nn.conv2d(self.input, w, strides=[1, stride, stride, 1], padding=padding)
            tf.summary.histogram('Conv',self.input)

            if bn is True:  # batch normalization
                self.input = self.BatchNorm(self.input,trainable)
            if bValue is not None:  # bias value
                b = self.ConstVariable(name='bias', shape=[outputChannel], value=bValue, trainable=trainable)
                self.input = tf.add(self.input, b)
            if sValue is not None:  # scale value
                s = self.ConstVariable(name='scale', shape=[outputChannel], value=sValue, trainable=trainable)
                self.input = tf.multiply(self.input, s)
            if activationFn is not None:  # activation function
                self.input = activationFn(self.input)
            tf.summary.histogram('Act',self.input)    
            tf.summary.histogram('weight',w)    
            tf.summary.histogram('bias',b)
        self.PrintLog(scope + ' output: ' + str(self.input.get_shape()))
        
    def MaxPool(self, k=2, s=None, globe=False):
        """
        Takes max value over a k x k area in each input map, or over the entire map (global = True)
        :param k: int
        :param globe:  int, whether to pool over each feature map in its entirety
        """
        self.count['mp'] += 1
        scope = 'maxpool_' + str(self.count['mp'])
        with tf.variable_scope(scope):
            if globe is True:  # Global Pool Parameters
                k1 = self.input.get_shape()[1]
                k2 = self.input.get_shape()[2]
                s1 = 1
                s2 = 1
                padding = 'VALID'
            else:
                k1 = k
                k2 = k
                if s is None:
                    s1 = k
                    s2 = k
                else:
                    s1 = s
                    s2 = s
                padding = 'SAME'
            # Max Pool Function
            self.input = tf.nn.max_pool(self.input, ksize=[1, k1, k2, 1], strides=[1, s1, s2, 1], padding=padding)
        self.PrintLog(scope + ' output: ' + str(self.input.get_shape()))

    def AvgPool(self, k=2, s=None, globe=False):
        """
        Averages the values over a k x k area in each input map, or over the entire map (global = True)
        :param k: int
        :param globe: int, whether to pool over each feature map in its entirety
        """
        self.count['ap'] += 1
        scope = 'avgpool_' + str(self.count['mp'])
        with tf.variable_scope(scope):
            if globe is True:  # Global Pool Parameters
                k1 = self.input.get_shape()[1]
                k2 = self.input.get_shape()[2]
                s1 = 1
                s2 = 1
                padding = 'VALID'
            else:
                k1 = k
                k2 = k
                if s is None:
                    s1 = k
                    s2 = k
                else:
                    s1 = s
                    s2 = s
                padding = 'SAME'
            # Average Pool Function
            self.input = tf.nn.avg_pool(self.input, ksize=[1, k1, k2, 1], strides=[1, s1, s2, 1], padding=padding)
            if globe is True:
                self.input = tf.reshape(self.input,[-1,self.input.get_shape()[3]])
        self.PrintLog(scope + ' output: ' + str(self.input.get_shape()))

    def Deconv2D(self, filterSize, outputChannels, stride=1, padding='SAME', activationFn=tf.nn.relu, bValue=0.0, sValue=1.0, bn=True, trainable=True):
        """
        2D Deconvolutional Layer
        :param filter_size: int. assumes square filter
        :param output_channels: int
        :param stride: int
        :param padding: 'VALID' or 'SAME'
        :param activation_fn: tf.nn function
        :param b_value: float
        :param s_value: float
        """
        self.count['deconv'] += 1
        scope = 'deconv_' + str(self.count['deconv'])
        with tf.variable_scope(scope):

            # Calculate the dimensions for deconv function
            batchSize = tf.shape(self.input)[0]
            inputHeight = tf.shape(self.input)[1]
            inputWidth = tf.shape(self.input)[2]

            if padding == "VALID":
                outRows = (inputHeight - 1) * stride + filterSize
                outCols = (inputWidth - 1) * stride + filterSize
            else:  # padding == "SAME":
                outRows = inputHeight * stride
                outCols = inputWidth * stride

            # Deconv function
            inputChannels = self.input.get_shape()[3]
            outputShape = [filterSize, filterSize, outputChannels, inputChannels]
            w = self.weight_variable(name='weights', shape=outputShape, trainable=trainable)
            deconv_out_shape = tf.stack([batchSize, outRows, outCols, outputChannels])
            self.input = tf.nn.conv2d_transpose(self.input, w, deconv_out_shape, [1, stride, stride, 1], padding)

            if bn is True:  # batch normalization
                self.input = self.batch_norm(self.input)
            if bValue is not None:  # bias value
                b = self.const_variable(name='bias', shape=[outputChannels], value=bValue, trainable=trainable)
                self.input = tf.add(self.input, b)
            if sValue is not None:  # scale value
                s = self.const_variable(name='scale', shape=[outputChannels], value=sValue, trainable=trainable)
                self.input = tf.multiply(self.input, s)
            if activationFn is not None:  # non-linear activation function
                self.input = activationFn(self.input)
        self.PrintLog(scope + ' output: ' + str(self.input.get_shape()))  # print shape of output
        
    def Flatten(self, keepProb=1):
        """
        Flattens 4D Tensor (from Conv Layer) into 2D Tensor (to FC Layer)
        :param keep_prob: int. set to 1 for no dropout
        """
        self.count['flat'] += 1
        scope = 'flat_' + str(self.count['flat'])
        with tf.variable_scope(scope):
            # Reshape function
            inputNodes = tf.Dimension(
                self.input.get_shape()[1] * self.input.get_shape()[2] * self.input.get_shape()[3])
            outputShape = tf.stack([-1, inputNodes])
            self.input = tf.reshape(self.input, outputShape)

            # Dropout function
            if keepProb != 1:
                self.input = tf.nn.dropout(self.input, keep_prob=keepProb)
        self.PrintLog(scope + ' output: ' + str(self.input.get_shape()))        

    def FC(self, outputNodes, keepProb=1, activationFn=tf.nn.relu, bValue=0.0, sValue=1.0, bn=True, trainable=True):
        """
        Fully Connected Layer
        :param output_nodes: int
        :param keep_prob: int. set to 1 for no dropout
        :param activation_fn: tf.nn function
        :param b_value: float or None
        :param s_value: float or None
        :param bn: bool
        """
        self.count['fc'] += 1
        scope = 'fc_' + str(self.count['fc'])
        with tf.variable_scope(scope):

            # Flatten if necessary
            if len(self.input.get_shape()) == 4:
                inputNodes = tf.Dimension(
                    self.input.get_shape()[1] * self.input.get_shape()[2] * self.input.get_shape()[3])
                outputShape = tf.stack([-1, inputNodes])
                self.input = tf.reshape(self.input, outputShape)

            # Matrix Multiplication Function
            inputNodes = self.input.get_shape()[1]
            outputShape = [inputNodes, outputNodes]
            w = self.weight_variable(name='weights', shape=outputShape, trainable=trainable)
            self.input = tf.matmul(self.input, w)

            if bn is True:  # batch normalization
                self.input = self.BatchNorm(self.input, 'fc')
            if bValue is not None:  # bias value
                b = self.const_variable(name='bias', shape=[outputNodes], value=bValue, trainable=trainable)
                self.input = tf.add(self.input, b)
            if sValue is not None:  # scale value
                s = self.const_variable(name='scale', shape=[outputNodes], value=sValue, trainable=trainable)
                self.input = tf.multiply(self.input, s)
            if activationFn is not None:  # activation function
                self.input = activationFn(self.input)
            if keepProb != 1:  # dropout function
                self.input = tf.nn.dropout(self.input, keep_prob=keepProb)
        self.PrintLog(scope + ' output: ' + str(self.input.get_shape()))        
        
    def BatchNorm(self,inputs,trainable=True):
        beta = self.ConstVariable(name='beta', shape=[inputs.get_shape()[-1]], value=0.0, trainable=trainable)
        gamma = self.ConstVariable(name='gamma', shape=[inputs.get_shape()[-1]], value=1.0, trainable=trainable)
        lens = len(inputs.get_shape())-1
        axises = np.arange(len(inputs.get_shape()) - 1)
        if lens == 1:
            axises = [0]
        elif lens == 3:
            axises = [0,1,2]                
        batchMean, batchVar = tf.nn.moments(inputs, axises, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        #滑动窗口来进行加权平均
        def MeanVarWithUpdate():
            emaApplyOp = ema.apply([batchMean, batchVar])
            with tf.control_dependencies([emaApplyOp]):
                return tf.identity(batchMean), tf.identity(batchVar)
        # 通过train_phase(一个布尔表达式，true选择执行接下来第一个，false选择执行后面一个)
        # 这里可以选择执行 正常均值，也可以选择用ExponentialMovingAverage
        mean, var = tf.cond(tf.cast(trainable,tf.bool), MeanVarWithUpdate, lambda: (ema.average(batchMean), ema.average(batchVar)))
        # Beta = 0 ,Gamma = 1,offset = 0.001
        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3,name='bn')
        tf.summary.histogram('Beta',beta)
        tf.summary.histogram('Gamma',gamma)
        return normed
    
    def get_output(self):
        """
        :return tf.Tensor, output of network
        """
        return self.input
    
    @staticmethod
    def PrintLog(message):
        """ Writes a message to terminal screen and logging file, if applicable"""
        print(message)
        logging.info(message)        
        
    @staticmethod
    def WeightVariable(name, shape, trainable):
        """
        :param name: string
        :param shape: 4D array
        :return: tf variable
        """
        w = tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(uniform=False), trainable=trainable, regularizer=tf.contrib.layers.l2_regularizer(1.0))
        return w

    @staticmethod
    def ConstVariable(name, shape, value, trainable):
        """
        :param name: string
        :param shape: 1D array
        :param value: float
        :return: tf variable
        """
        return tf.get_variable(name, shape, initializer=tf.constant_initializer(value), trainable=trainable)
    
    
    