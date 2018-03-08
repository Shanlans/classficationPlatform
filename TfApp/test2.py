# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 09:20:26 2018

@author: dxye
"""

"""
说明：cmd参数解析
示例：python test2.py --name dxye --age 25
"""

import argparse
import sys
import os

#parse = argparse.ArgumentParser()
#parse.add_argument('--name', type=str, default='zhangsan', help='your name')
#parse.add_argument('--age', type=int, default='18', help='your age')
#flags, unparsed = parse.parse_known_args(sys.argv[1:])
#print(flags.name)
#print(flags.age)
#print(unparsed)

#for dirName, subdirList, fileList in os.walk("d://pythonworkspace//DeepLearningPlatform//classficationPlatform//TfApp//test"):
#        print(dirName)
#        print(subdirList)
#        print(fileList)