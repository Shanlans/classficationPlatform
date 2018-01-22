# -*- coding: utf-8 -*-

# create by Shanlan on 22/1/2018

import tensorflow as tf


def AccuracyCal(labels,predicts):
    with tf.variable_scope('Acc'):
        correctPrediction = tf.equal(tf.argmax(predicts,1),tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(correctPrediction,tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    return accuracy
    
def ConfusionMatrix(labels,predicts):
    with tf.variable_scope('ConfusionMatrix'):
        c = tf.confusion_matrix(tf.argmax(labels,1),tf.argmax(predicts,1),2)
    return c
