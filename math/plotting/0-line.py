#!/usr/bin/env python3
'''
This script generates numbers between 0 and 10 for axes x and y,
but raises the ansers to the power of 3 for axis y.  From these numbers, it plots a line graph with red line
'''
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3
x = np.arange(0, 11)
'''Plotting the points'''
plt.plot(x, y, 'r-')  # Plot y as a red solid line
plt.xlim(0, 10)  # Set x-axis range from 0 to 10
plt.show()
