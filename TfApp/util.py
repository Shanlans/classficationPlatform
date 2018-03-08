# -*- coding: utf-8 -*-

import os

def MkDir(path):
    """
    Args:
        the path need to be created
    Returns:
        creating result
    """
    # eliminate first space
    path=path.strip()
    # eliminate '\\'
    path=path.rstrip("\\")
 
    # true is exists
    isExists=os.path.exists(path)
 
    if not isExists:
        os.makedirs(path)
        return True
    else:  
        return False
    
    
def ModelDictCreate(startpath):
    """
    Args:
        input a path
    Returns:
        folders contain model 
    """
    modelDict = {}
    # traverse the files and subdirs in the path
    for dirName, subdirList, fileList in os.walk(startpath,topdown=False):
        level = dirName.count(os.sep)
        if level > 0 :
            modelDict[dirName.split('\\')[-1]] = fileList
    return modelDict


