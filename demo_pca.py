# The demo of PCA algorithm

import numpy as np
import matplotlib.pyplot as plt

from ml.manifolds.unsupervised.pca import pca

# Load data
data = np.loadtxt("data/pcaData.txt").transpose()

# Display the size of loaded data
# print data.shape

# Visualizing 2D points
plt.scatter(data[:,0], data[:,1])

# Computing PCA
(U, s, mean) = pca.pca(data)
plt.scatter(mean[0], mean[1], c='r')

# Displaying basis vectors
ax = plt.axes()
ax.arrow(mean[0], mean[1], U[0,0]*np.sqrt(s[0]), U[1,0]*np.sqrt(s[0]), head_width=0.05, head_length=0.1, fc='k', ec='k')
ax.arrow(mean[0], mean[1], U[0,1]*np.sqrt(s[1]), U[1,1]*np.sqrt(s[1]), head_width=0.05, head_length=0.1, fc='k', ec='k')

plt.show()

