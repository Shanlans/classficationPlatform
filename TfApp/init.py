# -*- coding: utf-8 -*-
#Created on 15/1/2018 by Shen Shanlan


import os
import inputs
import scipy.ndimage
import tensorflow as tf
from tensorflow.python.client import device_lib

class Init(object):
    """
    Class Summary:
        initialize hyperparameters, processing parameters
        perepare training datasets
        
    Attributes:
        trainSwitch:
        babySitting: 
        learningRate:  
        trainBatchSize:
        trainStep:
        inputDataDir:
        dropout:   
        __trainPercent:
        __inputData:
        __maxValidateBatchSize:
        __train: 
        __validate:
        __test:  
        imageInfo: 
        imageScale:
        __inp:
        sess:
        xs:
        ys:
        __classes:
        classNum:    
    """  
    def __init__(self,
                 TRAIN_SWICH=True,
                 BABY_SITTING=True,                
                 LEARNING_RATE=1e-3,
                 TRAINING_BATCH_SIZE=64,
                 MAX_STEP=1000,
                 DATA_BASE=''): 
        """Initialize main input parameters
        Args:
            TRAIN_SWICH: Turn-on or Turn-off training progress
                        -True, Turn-on
                        -False,Turn-off
            BABY_SITTING: Restart training progress or Load pre-train model
                        -True, restart training progress
                        -False,load pre-train model
            LEARNING_RATE: Optimizer learning rate
            TRAINING_BATCH_SIZE: training mini batch size
            MAX_STEP: training iteration times
            DATA_BASE: input data dir
            classes: list of class name
        Returns:
            
        Raises:
            
        """
        self.trainSwitch    =   TRAIN_SWICH
        self.babySitting    =   BABY_SITTING
        self.learningRate   =   LEARNING_RATE
        self.trainBatchSize =   TRAINING_BATCH_SIZE
        self.trainStep      =   MAX_STEP
        self.inputDataDir   =   DATA_BASE        
        self.dropout        =   0.5       
        self.__trainPercent   =   70
        self.__inputData      =   {}
        
        self.__maxValidateBatchSize = 200
        self.__train          =   {}
        self.__validate       =   {}
        self.__test           =   {}
        self.imageInfo        =   {'imageInfoGet':False,'imageHeight':0,'imageWidth':0,'imageChannels':0}
        self.imageScale       =   4
        self.__inp            =   inputs.Input_Data()
        self.sess             =   tf.Session()
        self.xs               =   None
        self.ys               =   None
        self.__classes,self.classNum = self.__inp.GetClassNumber(self.inputDataDir)
#        local_device_protos = device_lib.list_local_devices()
#        self.gpuList = [x.name for x in local_device_protos if x.device_type == 'GPU']
        

         
        
        
    def LoadInputData(self,isShuffle=True,stage='Train'):         
        """load input data
        Load the training, validation and test data set from the path where the user gives or default setting
        Args:
            isShuffle: Shuffle dataset or not, True is shuffle
            stage: Genarate the batch for 'Train','Test' or 'Validate'
            Returns:
    
                Rasies:
        If No validation dataset, this function will randomly generate the validation dataset
        from training dataset by given percentage as the parameter "self.__trainPercent" setting    
        """
        
                
        trainPercent = self.__trainPercent
        dataBase = self.inputDataDir
        classes = self.__classes
        if stage is 'Train':
            dataBase = os.path.join(dataBase,'Train\\')
            data,label = self.__inp.GetFiles(dataBase,classes,isShuffle=isShuffle,stage='Train')
            self.__train['image'] = data
            self.__train['label'] = label
        elif stage is 'Validate':
            dataBase = os.path.join(dataBase,'Validate\\')
            if not os.listdir(dataBase):
                print('\nWarning: No validation dataset! Use %s%% training data as validation dataset.\n'%trainPercent)
                trainData = self.__inputData['trainDataSet']['image']
                trainLabel = self.__inputData['trainDataSet']['label']
                trainDataSlice,validateDataSlice=self.__inp.PercentListSlicing(trainData,trainPercent)
                trainLabelSlice,validateLabelSlice=self.__inp.PercentListSlicing(trainLabel,trainPercent)
                self.__train['image'] = trainDataSlice
                self.__train['label'] = trainLabelSlice
                self.__validate['image'] = validateDataSlice
                self.__validate['label'] = validateLabelSlice
            else:
                data,label = self.__inp.GetFiles(dataBase,classes,isShuffle=isShuffle,stage='Validate')
                self.__validate['image'] = data
                self.__validate['label'] = label
        elif stage is 'Test':
            dataBase = os.path.join(dataBase,'Test\\')
            data,label = self.__inp.GetFiles(dataBase,classes,isShuffle=isShuffle,stage='Test')
            self.__test['image'] = data
            self.__test['label'] = label   
        else:
            pass
                
        self.__inputData.update(trainDataSet=self.__train)
        self.__inputData.update(valiDataSet=self.__validate)
        self.__inputData.update(testDataSet=self.__test)
        
        if self.imageInfo['imageInfoGet'] is False:
            for k,v in self.__inputData.items():
                if 'image' in v.keys():
                    if len(v['image'])>0:
                        if len(scipy.ndimage.imread(v['image'][0]).shape) == 2:
                            height, width= scipy.ndimage.imread(v['image'][0]).shape
                            channels = 1
                        else:
                            height, width, channels = scipy.ndimage.imread(v['image'][0]).shape
                        self.imageInfo.update(imageInfoGet = True)
                        self.imageInfo.update(imageHeight = height*self.imageScale)
                        self.imageInfo.update(imageWidth = width*self.imageScale)
                        self.imageInfo.update(imageChannels = channels)
                        break
        
            
        
    def PrepareBatch(self,isShuffle=True,stage='Train'):
        classNum = self.classNum
        imageInfo = [self.imageInfo['imageHeight'],self.imageInfo['imageWidth'],self.imageInfo['imageChannels']]                       
        if stage is 'Train':
            batchSize = self.trainBatchSize
            data = self.__inputData['trainDataSet']['image']
            label = self.__inputData['trainDataSet']['label']
            numberThread = 2000
            isShuffle = True
            print('\nTraining batch %s ready.\n'%batchSize)
        elif stage is 'Validate':
            data = self.__inputData['valiDataSet']['image']
            label = self.__inputData['valiDataSet']['label']
            numberThread = 1
            isShuffle = False
            if len(label)<self.__maxValidateBatchSize:
                batchSize = len(label)
            else:
                batchSize = self.__maxValidateBatchSize               
            print('\nValidate batch %s ready.\n'%batchSize)
        elif stage is 'Test':
            data = self.__inputData['trainDataSet']['image']
            label = self.__inputData['trainDataSet']['label']
            batchSize = len(label)
            numberThread = 1 
            isShuffle = False
            print('\Test batch %s ready.\n'%batchSize)
        dataBatch,labelBatch=self.__inp.GetBatch(data,label,classNum,imageInfo[:],batchSize,isShuffle=isShuffle,numberThread=numberThread)
        
        return dataBatch,labelBatch
        
    def ClossSession(self):
        self.sess.close()
        print("Training Stop!!")
        
            
        
            
        