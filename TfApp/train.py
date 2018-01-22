# -*- coding: utf-8 -*-
# Created on 15/1/2018 by Shanlan Shen


import tensorflow as tf

import tensorflow.contrib.slim as slim

import init
import logsfilesfolders
import evaluation



class train(object):
    
    def __init__(self,
                 initial,
                 iteration,
                 learningRate,
                 optimizerMethod = 'Adam',                 
                 learningRateDecay = True,
                 learningRateDecayMethod = 'Exponential_decay',
                 learningRateDecayRate = 0.9,
                 regularization = False,
                 regularizationWeight = 0.01,
                 feedDict = {},
                 reportStep = 50,
                 checkPointStep = 2000,
                 fineTune = False,
                 sess 
                 ):
        
        self.__inital = initial
        self.__trainStep = iteration
        self.__optimizerMethod = optimizer
        self.__learningRate = learningRate
        self.__learningRateDecay = learningRateDecay
        self.__learningRateDecayMethod = learningRateDecayMethod
        self.__learningRateDecayRate = learningRateDecayRate
        self.__regularization = regularization
        self.__regularizationMethod = regularizationMethod
        self.__regularizationWeight = regularizationWeight
        self.__reportStep = reportStep
        self.__checkPointStep = checkPointStep
        self.__fineTune = fineTune
        self.__sess = sess         
        self.coord = tf.train.Coordinator()
        self.saver = tf.train.Saver()
        self.__feedDict = feedDict  
        self.__feedDict.updata(xs=None)
        self.__feedDict.updata(ys=None)        
        self.__initial.LoadInputData(stage='Train')
        self.__initial.LoadInputData(stage='Validate')
#        self.__initial.LoadInputData(stage='Test')
        self.__addForDecay

        imageInfo = [self.__initial.imageInfo['imageHeight'],self.__initial.imageInfo['imageWidth'],self.__initial.imageInfo['imageChannels']]  
        self.folder = logsfilesfolders.LogsFilesFolders('Train','V1',self.__sess)
        self.tb = self.folder.TbInital()
        with tf.name_scope('Inputs'):
            self.xs = tf.placeholder(tf.float32,imageInfo.insert(0,None),'Images')
            self.ys = tf.placeholder(tf.float32,[None,self.__inital.classNum],'Labels')
              
        
    
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
            
        
    def __Oprimizer(self,loss): 
        if self.__learningRateDecay:
            with tf.control_dependencies([self.__addForDecay]):      
                if self.__optimizerMethod is 'Adam':
                    with tf.variable_scope('Adam'):
                        trainOps = tf.train.AdamOptimizer(self.__learningRate).minimize(loss)
        else:
            if self.__optimizerMethod is 'Adam':
                    with tf.variable_scope('Adam'):
                        trainOps = tf.train.AdamOptimizer(self.__learningRate).minimize(loss)
        return trainOps 
    
    def __DataFeed(self,stage='Train'):        
        datas,lables = self.__sess.run([self.__inital.PrepareBatch(stage=stage)])
        self.__feedDict['xs'] = datas
        self.__feedDict['ys'] = lables 
    
        if stage is 'Train':
            if 'dropout' in self.__feedDict.keys():
                self.__feedDict['dropout'] = self.__inital.dropout
            if 'trainPhase' in self.__feedDict.keys():
                self.__feedDict['trainPhase'] = True                
        else:
            if 'dropout' in self.__feedDict.keys():
                self.__feedDict['dropout'] = 1
                self.__feedDict['trainPhase'] = False
                
    def __Backward(self,predicts,step):
        self.__DataFeed(stage='Train')
        lossOps = self.__LossCal(predicts)
        trainOps= self.__Optimizer(lossOps)
        accOps = evaluation.AccuracyCal(self.ys,predicts)        
        _,loss,acc = self.__sess.run([trainOps,lossOps,accOps],feed_dict=self.__feedDict)
        if step % self.__reportStep == 0:            
            print('Step %d, learning rate = %s'%(step,self.__sess.run(self.__learningRate)))
            print('Step %d, train loss = %.4f, train accuracy = %.4f%%' %(step, loss, acc*100.0))
            summary = sess.run(self.tb[0],feed_dict=self.__feedDict)
            self.tb[1].add_summary(summary, step)
        if step % self.__checkPointStep == 0:     
            self.saver.save(sess, self.folder.mainModelDir, global_step=step)
            
            
    def __forward(self,predict,step):       
        if step % self.__reportStep == 0:
            self.__DataFeed(stage='Validate')
            lossOps = self.__LossCal(predicts)
            accOps = evaluation.AccuracyCal(self.ys,predicts)
            cMatOps= evaluation.ConfusionMatrix(self.ys,predicts)
            loss,acc,cMat = self.__sess.run([lossOps,accOps,cMatOps],feed_dict=self.__feedDict)            
            print('Step %d, validate loss = %.4f, validate accuracy = %.4f%%' %(step, loss, acc*100.0))
            print('The Confusion matrix = \n%s'%(cMat)) 
            summary = sess.run(self.tb[0],feed_dict=self.__feedDict)
            self.tb[2].add_summary(summary, step)

    def TrainProcess(self):       
        '''
           Args:
           Return:
            
        '''
        __LearningRateDecay()        
        init = tf.global_variables_initializer()
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)
        threads = tf.train.start_queue_runners(sess=self.__sess, coord=self.coord)
                                
        try:
            for step in np.arange(self.__trainStep):
                if self.coord.should_stop():
                    break              
                self.__Backward(predicts,step)                 
                self.__forward(predicts,step)
                                   
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        
        coord.join(threads)    
                
                
            
            
        
        
        
        
