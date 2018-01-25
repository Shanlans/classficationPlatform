#import tensorflow as tf
#w = tf.Variable(1.0)
#ema = tf.train.ExponentialMovingAverage(0.9)
#update = tf.assign_add(w, 1.0)
#
#ema_op = ema.apply([update])
#with tf.control_dependencies([ema_op]):
#    ema_val = tf.identity(ema.average(update))
#
#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
#for i in range(3):
#    print(sess.run([ema_val]))
#sess.close()

import tensorflow as tf
import cv2
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


print(len(get_available_gpus()))