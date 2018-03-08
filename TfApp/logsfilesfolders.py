# -*- coding: utf-8 -*-
# Created on 10/1/2018 by Shen Shanlan

import os
import time
import util
import tensorflow as tf
from tensorflow.python.framework import graph_util 

class LogsFilesFolders(object):
    """
    Class Summary:
        create folders for logs('event' files which can be loaded in tensorboard)
        combine graph and parameters to 'pb' files
        initialize tensorboard
        
    Attributes:
        folderVersion:version number
        folderPurpose:the folder is for train, validate or test
        __DATE:used to specify every training or testing
        __modelDir:model path
        __logDir:log path
        __sess:session
        modelDict:
        mainModelDir:    
        mainLogDir:    
    """
    def __init__(self,
                 stage,
                 folderVersion,
                 sess
                 ):

        self.folderVersion = folderVersion
        self.folderPurpose = stage
        self.__DATE = time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))
        self.__modelDir = "./model"
        self.__logDir = "./Logs"
        self.modelDict = {}
        self.__sess = sess
        self.mainModelDir = os.path.join(self.__modelDir,self.folderVersion,self.__DATE) #should return
        self.mainLogDir = os.path.join(self.__logDir,self.folderVersion,self.__DATE)
        if self.folderPurpose is "Train":
            util.MkDir(os.path.join(self.mainModelDir))
            util.MkDir(os.path.join(self.mainLogDir,'Train'))
            util.MkDir(os.path.join(self.mainLogDir,'Validate'))
        elif self.folderPurpose is "Test":
            util.MkDir(os.path.join(self.mainLogDir,'Test'))
            # Get the existed model list on the specified data and version for testing
            self.modelDict = util.ModelDictCreate(os.path.join(self.__modelDir,self.folderVersion)) 
        
                    
    def GeneratePb(self,modelFolder=None,ckptNum=4999,sess=None):  
        # retrieve our checkpoint fullpath
        if modelFolder is None:
            modelFolder = self.mainModelDir
        
        if sess is None:
            sess = self.__sess
        
        ckptName = 'IR.ckpt'+'-'+str(ckptNum)
        pbName = 'IR'+'-'+str(ckptNum)+'.pb'
        
        checkPoint = tf.train.get_checkpoint_state(modelFolder,"checkpoint")                
        ckptList = [i.split("\\")[-1] for i in checkPoint.all_model_checkpoint_paths]
        
        if ckptName in ckptList:
            print('Check point: %s founded!'%ckptName)
        else:
            print('Check point not exist, check again')
            return 
        loadModel = os.path.join(modelFolder,ckptName)
        graphName = loadModel + '.meta'
        pbModel = os.path.join(modelFolder,pbName)
        outputNode= "Softmax/Softmax"
        saver = tf.train.import_meta_graph(graphName, clear_devices=True)  
        graph = tf.get_default_graph()  
        inputGraphDef = graph.as_graph_def() 
        saver.restore(sess, loadModel)
        outputGraphDef = graph_util.convert_variables_to_constants(  
                sess,   
                inputGraphDef,   
                outputNode.split(",") # We split on comma for convenience  
            )
        with tf.gfile.GFile(pbModel, "wb") as f:
                f.write(outputGraphDef.SerializeToString())  
        print("%d ops in the final graph." % len(outputGraphDef.node))  
        
    def TbInital(self,logsFolder=None,sess=None):
        if self.folderPurpose is 'Train':
            if logsFolder is None:
                logsFolder = self.mainLogDir
                
            if sess is None:
                sess = self.__sess
            
            trainLogDir = os.path.join(logsFolder,'Train')
            validateLogDir = os.path.join(logsFolder,'Validate')
            
            merged = tf.summary.merge_all()    
            trainWriter = tf.summary.FileWriter(trainLogDir,sess.graph)  
            validateWriter = tf.summary.FileWriter(validateLogDir,sess.graph) 
            return [merged,trainWriter,validateWriter]
        
        
        
                

            
        