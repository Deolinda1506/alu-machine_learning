#!/usr/bin/env python3
'''calculate the Loss'''

import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss

    Args:
    y: placeholder - true labels for the input data
    y_pred: tensor - network’s predictions

    Returns:
    tensor - loss of the predictions
    """
    # Compute the softmax cross-entropy loss
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=y, logits=y_pred)
    return loss
