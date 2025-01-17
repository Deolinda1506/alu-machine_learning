#!/usr/bin/env python3
'''
This script generates random height and weight values 2000 of them with a seed of 5
based on the mean and covariance , then transposes it before plotting it on a scatter graph
'''
import numpy as np
import matplotlib.pyplot as plt

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

'''Plotting the scatter plots'''

plt.scatter(x,y,color='magenta')
plt.xlabel('Height (in)')
plt.ylabel('Weight (lbs)')
plt.title('Men\'s Height vs Weight')
plt.show()

