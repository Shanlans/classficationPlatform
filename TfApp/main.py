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


def main(_):      
    initial = init.Init(trainSwitch,
                        babysitting,
                        para["learning_rate"],
                        para["mini_batch_size"],
                        para["max_steps"],
                        os.path.join(os.getenv('DATADIR', 'Database'), para["data_dir"])
                        )
  
    trainInstance = train.Train(initial,filterSizes=[3,3,1,1],featureNums=[20,40,40,initial.classNum])
    trainInstance.TrainProcess()
    initial.ClossSession()

if __name__ == '__main__':
#  read para from console

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

#  read para from txt

  with open('para.txt', encoding='utf-8') as f:
      para = json.load(f)
  if para["trainSwitch"] == "True":
      trainSwitch = True
  else:
      trainSwitch = False
  if para["babysitting"] == "True":
      babysitting = True
  else:
      babysitting = False 
  tf.app.run(main=main)
