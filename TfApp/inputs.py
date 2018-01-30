# -*- coding: utf-8 -*-
# Created by Shanlan Shen on 8/10/2017

import tensorflow as tf
import numpy as np
import os

class Input_Data(object): 
    
    def GetFiles(self,fileDir,classes,isShuffle=True,stage='Train'):        
        classNumber=len(classes)       
        fileNameDict = {}
        fileLabelDict = {}       
        for c in range(classNumber):
            fileNameDict[c] = []
            fileLabelDict[c] = []
        
    
        for file in os.listdir(fileDir):
            name = file.split(sep='_')
            for n in range(classNumber):
                className = classes[n]
                if name[0] == className:
                    fileNameDict[n].append(fileDir+file)
                    fileLabelDict[n].append(n)
                    break
                else:
                    continue
                
        print("{} stage has:".format(stage))
        for n in range(classNumber):
            sampleNum = len(fileLabelDict[n])
            className = classes[n]
            print("    Class %d: %s = %d"%(n,className,sampleNum))
            
        imageList = np.hstack( fileNameDict[k] for k in sorted(fileNameDict))
        labelList = np.hstack( fileLabelDict[k] for k in sorted(fileLabelDict))
        
        temp = np.array([imageList,labelList])
        temp = temp.transpose()
        
        if isShuffle:
            np.random.shuffle(temp)
            print("    Data shuffled!!!  ")
        else:
            print("    Data in Order!!!  ")
            pass
        
        imageList = list(temp[:,0])
        labelList = list(temp[:,1])
        labelList = [int(i) for i in labelList]  
        return imageList, labelList         


    def GetBatch(self,image,label,classNum,imageInfo,batchSize,isShuffle=True,numberThread=1000,capacity=5000):
        '''
        Args:
            image: list type
            label: list type
            classNum: Classification class number
            imageInfo: A list [imageHight,imageWidth,imageChannels]
            batchSize: batch size
            isShuffle: True, shuffle batch;False, no shuffle batch
            capacity: the maximum elements in queue
            numberThread: if 1, no shuffle, if >1 shuffle number
        Return:
            image batch: 4D tensor [batch_size, image_W, image_H, image_Channel],dtype=tf.float32
            label_batch: 1D tensor [batch_size], dtype=tf.float32
        '''
        imageHight,imageWidth,imageChannels = imageInfo
        label=tf.one_hot(label,classNum,1,0,-1,dtype=tf.int32)
        image = tf.cast(image,tf.string)
        input_queue = tf.train.slice_input_producer([image,label],shuffle=isShuffle)
       
        label =input_queue[1]
        image_contents = tf.read_file(input_queue[0])
        image = tf.image.decode_png(image_contents,channels=imageChannels)
        image = tf.image.resize_images(image,[imageHight,imageWidth],method=3)        
        image = tf.image.per_image_standardization(image)
        image_batch, label_batch = tf.train.batch([image, label],
                                                  batch_size= batchSize,
                                                  num_threads= numberThread, 
                                                  capacity = capacity)
              
        label_batch = tf.reshape(label_batch, [batchSize,classNum])
        
        image_batch = tf.cast(image_batch, tf.float32)
        return image_batch, label_batch
        
        
    def PercentListSlicing(self,l,p):
        percentage = float(p)/100.0
        theList = l[:int(len(l)*percentage)]
        anotherList = l[int(len(l)*percentage):]
        return theList,anotherList
    
