#!/usr/bin/env python3
'''Accuracy'''
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of predictions
    Args:
    y: placeholder - true labels for the input data
    y_pred: tensor - network’s predictions
    Returns:
    tensor - decimal accuracy of the predictions
    """
    # Calculate the correct predictions
    correct_predictions = tf.equal(tf.argmax
                                   (y_pred, 1), tf.argmax(y, 1))

    # Calculate the accuracy by via the mean of correct predictions
    accuracy = tf.reduce_mean(tf.cast
                              (correct_predictions, tf.float32))
    return accuracy
