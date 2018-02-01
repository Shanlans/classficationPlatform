#Tensorflow Main function
#Shenshanlan Created on 2017/12/26

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tensorflow as tf 

import init
import train




FLAGS = None


def main(_):    
  initial = init.Init(FLAGS.trainSwitch,
                      FLAGS.babysitting,
                      FLAGS.learning_rate,
                      FLAGS.mini_batch_size,
                      FLAGS.max_steps,
                      FLAGS.data_dir,
                      classes = ['Dianzi','Zangwu','Others'] 
                      )

  trainInstance = train.Train(initial,200,0.0001,checkPointStep=10)
  trainInstance.TrainProcess()
  initial.ClossSession()
  
#  print(folder.modelDict['2018-01-10-15-39'])
#  folder.generatePb('./model/V1/2018-01-10-15-39')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--trainSwitch',type=bool,default=True,help='Swich on/off training')
  parser.add_argument('--babysitting',type=bool,default=True,help='Babysitting or Fine tuning')
  parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--mini_batch_size', type=int, default=16,
                      help='Training batch size')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default=os.path.join(os.getenv('DATADIR', 'Database'),
                           'IR_data'),
      help='Input data')
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('LOGDIR', 'Logs'),
                           'IR_log'),
      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

