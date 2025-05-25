#!/usr/bin/env python3
'''
That returns two placeholders, x and y, for the neural network
'''

import tensorflow as tf


def create_placeholders(nx, classes):
    '''
    Returns two placeholders, x and y, for the neural network.

    Args:
    nx: the number of feature columns in the input data
    classes: the number of output classes in the classifier

    Returns:
    x: placeholder for input data (shape: [None, nx])
    y: placeholder for labels (one-hot encoded, shape: [None, classes])
    '''
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y
