# -*- coding: utf-8 -*-
# Created on 15/1/2018 by Shanlan Shen


import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

import models

#import init
import logsfilesfolders
import evaluation



class Train(object):
    
    def __init__(self,
                 initial,
                 iteration,
                 learningRate,
                 optimizerMethod = 'Adam',                 
                 learningRateDecay = False,
                 learningRateDecayMethod = 'Exponential_decay',
                 learningRateDecayRate = 0.9,
                 regularization = False,
                 regularizationWeight = 0.01,
                 feedDict = {},
                 reportStep = 50,
                 checkPointStep = 2000,
                 fineTune = False
                 ):
        
        self.__initial = initial
        self.__trainStep = iteration
        self.__optimizerMethod = optimizerMethod
        self.__learningRate = learningRate
        self.__learningRateDecay = learningRateDecay
        self.__learningRateDecayMethod = learningRateDecayMethod
        self.__learningRateDecayRate = learningRateDecayRate
        self.__regularization = regularization
        self.__regularizationWeight = regularizationWeight
        self.__reportStep = reportStep
        self.__checkPointStep = checkPointStep
        self.__fineTune = fineTune
        self.__sess = self.__initial.sess         
        self.__coord = tf.train.Coordinator()
        self.__feedDict = feedDict                
        self.__initial.LoadInputData(stage='Train')
        self.__initial.LoadInputData(stage='Validate')
        self.__addForDecay = None

        inputInfo = [None,self.__initial.imageInfo['imageHeight'],self.__initial.imageInfo['imageWidth'],self.__initial.imageInfo['imageChannels']] 
        self.folder = logsfilesfolders.LogsFilesFolders('Train','V1',self.__sess)
        
        with tf.name_scope('Inputs'):
            self.xs = tf.placeholder(tf.float32,inputInfo,'Images')
            self.ys = tf.placeholder(tf.float32,[None,self.__initial.classNum],'Labels')
        self.dropout = tf.placeholder(tf.float32,name='Dropout')
        self.trainphase = tf.placeholder(tf.bool,name='Trainphase')
                    
        self.models = models.models(self.xs,[3,3,1,1],[20,40,40,self.__initial.classNum],trainable=True)
        
              
        
    
    def __LossCal(self,predicts):
        regularization = self.__regularization
        regularizationWeight = self.__regularizationWeight
        
        with tf.variable_scope('Loss'):
            loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.ys,logits=predicts,weights=1))
            regularizationVariable = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        
            if regularization is True:           
                loss += tf.multiply(regularizationWeight,sum(regularizationVariable))  
            else:
                regularizationVariable = None
            tf.summary.scalar('CrossEntropy',loss)
        return loss
        
    def __LearningRateDecay(self):  
        if self.__learningRateDecay:
            with tf.variable_scope('LRDecay'):
                initialStep = tf.get_variable('DecayStep',dtype=tf.int32,initializer=tf.constant(0),trainable=False)     
#                initialStep = tf.Variable(0, trainable=False)
                self.__learningRate = tf.train.exponential_decay(self.__learningRate,
                                                                 global_step=initialStep,
                                                                 decay_steps=500,
                                                                 decay_rate=self.__learningRateDecayRate)   

            self.__addForDecay = initialStep.assign_add(1)
            
        
    def __Optimizer(self): 
        self.__lossOps = self.__LossCal(self.__predicts)
        if self.__learningRateDecay:
            with tf.control_dependencies([self.__addForDecay]):      
                if self.__optimizerMethod is 'Adam':
                    with tf.variable_scope('Adam'):
                        self.__trainOps = tf.train.AdamOptimizer(self.__learningRate).minimize(self.__lossOps)
        else:
            if self.__optimizerMethod is 'Adam':
                    with tf.variable_scope('Adam'):
                        self.__trainOps = tf.train.AdamOptimizer(self.__learningRate).minimize(self.__lossOps)
    

        
    def __DataFeed(self,dataOps,labelOps,stage='Train'): 
        datas,lables = self.__sess.run([dataOps,labelOps])      
        self.__feedDict[self.xs] = datas
        self.__feedDict[self.ys] = lables 
        
        if stage == 'Train':
            self.__feedDict[self.dropout] = 0.5
            self.__feedDict[self.trainphase] = True
        else:
            self.__feedDict[self.dropout] = 1
            self.__feedDict[self.trainphase] = False
    
            
    def __Backward(self,step):
        self.__DataFeed(self.__trainDataOps,self.__trainLabelOps,stage='Train')
        accOps = evaluation.AccuracyCal(self.ys,self.__predicts)        
        _,loss,acc = self.__sess.run([self.__trainOps,self.__lossOps,accOps],feed_dict=self.__feedDict)
        if step % self.__reportStep == 0:            
            print('Step %d, learning rate = %s'%(step,self.__learningRate))
            print('Step %d, train loss = %.4f, train accuracy = %.4f%%' %(step, loss, acc*100.0))
            summary = self.__sess.run(self.tb[0],feed_dict=self.__feedDict)
            self.tb[1].add_summary(summary, step)
        if step % self.__checkPointStep == 0:
            self.__saver.save(self.__sess,self.folder.mainModelDir+'\\', global_step=step)
            
            
    def __Forward(self,step):       
        if step % self.__reportStep == 0:
            self.__DataFeed(self.__validateDataops,self.__validateLabelOps,stage='Validate')
            accOps = evaluation.AccuracyCal(self.ys,self.__predicts)
            cMatOps= evaluation.ConfusionMatrix(self.ys,self.__predicts)
            loss,acc,cMat = self.__sess.run([self.__lossOps,accOps,cMatOps],feed_dict=self.__feedDict)            
            print('Step %d, validate loss = %.4f, validate accuracy = %.4f%%' %(step, loss, acc*100.0))
            print('The Confusion matrix = \n%s'%(cMat)) 
            summary = self.__sess.run(self.tb[0],feed_dict=self.__feedDict)
            self.tb[2].add_summary(summary, step)
            

     
    def __BatchPrepare(self):
        self.__trainDataOps,self.__trainLabelOps=self.__initial.PrepareBatch(stage='Train')
        self.__validateDataops,self.__validateLabelOps = self.__initial.PrepareBatch(stage='Validate') 
        
    def __SetUpGraphic(self):
        self.__predicts = self.models.get_output()
        self.__Optimizer()
        self.__LearningRateDecay()
        self.__saver = tf.train.Saver()
        self.tb = self.folder.TbInital()
        slim.model_analyzer.analyze_vars(tf.trainable_variables(), print_info=True)
        
    def __InitialParameter(self):
        self.__threads = tf.train.start_queue_runners(sess=self.__sess, coord=self.__coord)   
        self.__sess.run(tf.global_variables_initializer())

    def TrainProcess(self):       
        '''
           Args:
           Return:
            
        '''
        self.__BatchPrepare()
        self.__SetUpGraphic()
        self.__InitialParameter()
          
        
        try:
            for step in np.arange(self.__trainStep):
                if self.__coord.should_stop():
                    break              
                self.__Backward(step)                 
                self.__Forward(step)
                                   
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            self.__coord.request_stop()
            
        self.__coord.join(self.__threads) 
           
                
                
            
            
        
        
        
        
