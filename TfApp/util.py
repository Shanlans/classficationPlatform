# -*- coding: utf-8 -*-

import os

def MkDir(path):
    # 引入模块 
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
 
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
 
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在        
        return False
    
    
def ModelDictCreate(startpath):
    modelDict = {}
    for dirName, subdirList, fileList in os.walk(startpath,topdown=False):
        level = dirName.count(os.sep)
        if level > 0 :
            modelDict[dirName.split('\\')[-1]] = fileList
    return modelDict


