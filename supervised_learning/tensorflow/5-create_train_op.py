#!/usr/bin/env python3
'''Train the OP'''

import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network
    Args:
    loss: tensor - loss of the network’s prediction
    alpha: float - learning rate
    Returns:
    train_op: operation -
    operation that trains the network using gradient descent
    """
    # Create a GradientDescentOptimizer object
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=alpha)

    # Create the training operation
    train_op = optimizer.minimize(loss)

    return train_op
