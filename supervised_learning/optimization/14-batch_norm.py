#!/usr/bin/env python3
# Function that creates a batch normalization layer for a neural network

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for
    a neural network in TensorFlow.

    Args:
        prev: Activated output of the previous layer
        n: Number of nodes in the layer to be created
        activation: Activation function to use on the output

    Returns:
        Activated output of the layer
    """
    init = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg")
    dense = tf.keras.layers.Dense(units=n, kernel_initializer=init)
    Z = dense(prev)
    mean, variance = tf.nn.moments(Z, axes=[0])
    beta = tf.Variable(tf.zeros([n]), trainable=True)
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    Z_norm = tf.nn.batch_normalization(Z, mean, variance, beta, gamma, 1e-8)
    return activation(Z_norm)
