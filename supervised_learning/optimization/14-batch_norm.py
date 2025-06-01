#!/usr/bin/env python3
"""
Creates a batch normalization layer for a neural network in TensorFlow 1.x
"""

import tensorflow as tf

def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer

    Parameters:
    - prev: activated output of the previous layer
    - n: number of nodes in the layer to be created
    - activation: activation function to use

    Returns:
    - Tensor of the activated output for the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    
    # Dense layer without activation
    dense = tf.layers.Dense(units=n, kernel_initializer=init)(prev)

    # Batch normalization parameters
    mean, variance = tf.nn.moments(dense, axes=[0])
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)

    # Batch normalization
    normalized = tf.nn.batch_normalization(dense, mean, variance, beta, gamma, 1e-8)

    # Activation
    return activation(normalized)
