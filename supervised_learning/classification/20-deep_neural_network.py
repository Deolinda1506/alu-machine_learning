#!/usr/bin/env python3
"""Defines a deep neural network performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network for binary classification"""

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network

        Parameters:
        nx (int): number of input features
        layers (list): list of nodes in each layer

        Raises:
        TypeError, ValueError: for invalid nx or layers
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(1, self.L + 1):
            layer_input_size = nx if l == 1 else layers[l - 2]
            self.weights[f"W{l}"] = (
                np.random.randn(layers[l - 1], layer_input_size) *
                np.sqrt(2 / layer_input_size)
            )
            self.weights[f"b{l}"] = np.zeros((layers[l - 1], 1))

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network

        Parameters:
        X (np.ndarray): shape (nx, m), input data

        Returns:
        tuple: output of the neural network and the cache
        """
        self.cache["A0"] = X

        for l in range(1, self.L + 1):
            W = self.weights[f"W{l}"]
            b = self.weights[f"b{l}"]
            A_prev = self.cache[f"A{l - 1}"]
            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))  # Sigmoid activation
            self.cache[f"A{l}"] = A

        return A, self.cache

    def cost(self, Y, A):
        """
        Calculates the cost using logistic regression

        Parameters:
        Y (np.ndarray): true labels, shape (1, m)
        A (np.ndarray): activated output, shape (1, m)

        Returns:
        float: logistic regression cost
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions

        Parameters:
        X (np.ndarray): shape (nx, m), input data
        Y (np.ndarray): shape (1, m), correct labels

        Returns:
        tuple: prediction (np.ndarray), cost (float)
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost
