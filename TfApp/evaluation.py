# -*- coding: utf-8 -*-

# create by Shanlan on 22/1/2018

import tensorflow as tf


def AccuracyCal(labels,predicts):
    """Calculate the accuracy
    Args:
        labels:
        predicts:
    
    Returns:
        Accuracy.
    
    """
    correctPrediction = tf.equal(tf.argmax(predicts,1),tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction,tf.float32))        
    return accuracy
    
def ConfusionMatrix(labels,predicts,classNumber):
    """Calculate the confusion matrix
    Args:
        labels:
        predicts:
        classNumber:
    
    Returns:
        Confusion matrix.
        
    Examples:
        
        ```
        c = [[98 2]
             [1 99]]
        ```
    
    """
    c = tf.confusion_matrix(tf.argmax(labels,1),tf.argmax(predicts,1),classNumber)
    return c

