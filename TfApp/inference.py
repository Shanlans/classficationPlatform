# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 18:21:22 2018

@author: dxye
"""

import logsfilesfolders

class Inference(object):
    
    def __init__(self,
                 modelDetailDir,
                 ckptNum,
                 sess,
                 ):
        self.__modelDir = './model' + modelDetailDir
        self.__sess = sess
        self.model = GeneratePb(self.__modelDir, ckptNum-1, self.__sess)
        
    def forward(self,
                dataDir,
                ):
        