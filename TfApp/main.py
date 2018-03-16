#Tensorflow Main function
#Shenshanlan Created on 2017/12/26

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import json
import tensorflow as tf 

import init
import train

FLAGS = None


def main():      
    initial = init.Init(para["trainSwitch"],
                        para["babysitting"],
                        para["learning_rate"],
                        para["mini_batch_size"],
                        para["max_steps"],
                        os.path.join(os.getenv('DATADIR', 'Database'), para["data_dir"])
                        )
  
    trainInstance = train.Train(initial,filterSizes=[3,3,1,1],featureNums=[20,40,40,initial.classNum])
    trainInstance.TrainProcess()
    initial.ClossSession()

def singleTrain(trainSwitch, babySitting, learningRate, batchSize, maxSteps, dataPath):
    """Train for one time"""
    initial = init.Init(trainSwitch,
                        babySitting,
                        learningRate,
                        batchSize,
                        maxSteps,
                        os.path.join(os.getenv('DATADIR', 'Database'), dataPath)
                        )
    trainInstance = train.Train(initial,filterSizes=[3,3,1,1],featureNums=[20,40,40,initial.classNum])
    trainInstance.TrainProcess()
    initial.ClossSession()
    
def multiTrain():
    with open('para.txt', encoding='utf-8') as f:
        """Train for several times according to the parameter file"""
        paras = json.load(f)
        paraNum = len(paras)
        
        initial = init.Init(paras[0]["trainSwitch"],
                            paras[0]["babysitting"],
                            paras[0]["learning_rate"],
                            paras[0]["mini_batch_size"],
                            paras[0]["max_steps"],
                            os.path.join(os.getenv('DATADIR', 'Database'),paras[0]["data_dir"])
                            )
#        trainInstance = train.Train(initial,
#                                    filterSizes=[3,3,1,1],
#                                    featureNums=[20,40,40,initial.classNum]
#                                    )
        
        lossList = []
        accList = []
        
        for trainIndex in range(paraNum):
            print('\n--------------------------------------------------------\n')
            print('Multi-Train-Stage-',trainIndex)
            print('\n--------------------------------------------------------\n')
            
            initial.trainSwitch = paras[trainIndex]["trainSwitch"]
            initial.babySitting = paras[trainIndex]["babysitting"]
            initial.learningRate = paras[trainIndex]["learning_rate"]
            initial.trainBatchSize = paras[trainIndex]["mini_batch_size"]
            initial.trainStep = paras[trainIndex]["max_steps"]
            initial.inputDataDir = os.path.join(os.getenv('DATADIR', 'Database'),paras[trainIndex]["data_dir"])

            trainInstance = train.Train(initial,
                                    filterSizes=[3,3,1,1],
                                    featureNums=[20,40,40,initial.classNum]
                                    )

            loss, acc = trainInstance.TrainProcess()
            lossList.append(loss)
            accList.append(acc)
            
            tf.reset_default_graph()
            initial.ClossSession()
            initial.sess = tf.Session()
#            trainInstance.updatemodel()      

        initial.ClossSession()
        print('\n--------------------------------------------------------\n')
        print('Training loss:\n')
        print(lossList)
        print('\nTraining accuracy:\n')
        print(accList)
    

if __name__ == '__main__':
###  read para from console

#  parser = argparse.ArgumentParser()
#  parser.add_argument('--trainSwitch',type=bool,default=True,help='Switch on/off training')
#  parser.add_argument('--babysitting',type=bool,default=True,help='Babysitting or Fine tuning')
#  parser.add_argument('--max_steps', type=int, default=200,
#                      help='Number of steps to run trainer.')
#  parser.add_argument('--learning_rate', type=float, default=0.001,
#                      help='Initial learning rate')
#  parser.add_argument('--mini_batch_size', type=int, default=16,
#                      help='Training batch size')
#  parser.add_argument(
#      '--data_dir',
#      type=str,
#      default=os.path.join(os.getenv('DATADIR', 'Database'),
#                           'IR_data'), 
#      help='Input data')
#
#  FLAGS, unparsed = parser.parse_known_args()
#tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

###  read para from txt

  with open('para.txt', encoding='utf-8') as f:
      para = json.load(f)
  tf.app.run(main=main)
