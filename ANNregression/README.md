[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **ANNregression** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of Quantlet: ANNregression

Published in: Quantlet

Description: 'Gradient descent implementation from scratch in order to train an artificial neural network to fit a regression of city-cycle fuel consumption in miles per gallon.'

Keywords: Dense, ANN, MLP, deep learning, neural network

Author: Bruno Spilak

Submitted:  2019-02-05 by Bruno Spilak

Output:
- ANNregression1.png

Data:
- mpg.csv
```

![Picture1](ANNregression1.png)

### PYTHON Code
```python

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# cf: https://github.com/antoine-eon/olga/tree/master/Algorithme/Algo%20Reseau%20Neuronal%20Regression%20Lineaire%20EX

# Load the data and create the data matrices X and Y
# This creates a feature vector X with a column of ones (bias)
# and a column of car weights.
# The target vector Y is a column of MPG values for each car.

X_file = np.genfromtxt('mpg.csv', delimiter=',', skip_header=1)
N = np.shape(X_file)[0]
X = np.hstack((np.ones(N).reshape(N, 1), X_file[:, 4].reshape(N, 1)))
Y = X_file[:, 0]

# Standardization
X[:, 1] = (X[:, 1]-np.mean(X[:, 1]))/np.std(X[:, 1])

# Two weights (bias and feature)
w = np.array([0, 0])

# Batch gradient descent
# size eta
max_iter = 100
eta = 1e-4
for t in range(0, max_iter):
    print(t)
    # We iterate over each data point for one epoch
    grad_t = np.array([0., 0.])
    for i in range(0, N):
        x_i = X[i, :]
        y_i = Y[i]
        h = np.dot(w, x_i)-y_i
        grad_t += 2*x_i*h
    # Update the weights
    w = w - eta*grad_t
    
    # Plot the data and best fit line
    tt = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 10)
    bf_line = w[0]+w[1]*tt
    plt.plot(X[:, 1], Y, 'kx', tt, bf_line)#, label = t)

# Plot the data and best fit line
plt.plot(X[:, 1], Y, 'kx', tt, bf_line, 'r-')
plt.savefig('figure1.png')

plt.show()
print("Weights found:",w)



```

automatically created on 2019-03-06