# The demo of PCA algorithm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab import PCA

from ml.manifolds.unsupervised.pca import pca
from misc import zero_mean as zm

# Load data
data = np.loadtxt("data/pcaData.txt").transpose()

data,mean = zm.zero_mean_common(data)
# Display the size of loaded data
# print data.shape

# Visualizing 2D points
plt.figure(1)
plt.scatter(data[:,0], data[:,1])

# Computing PCA
(U, s, mean) = pca.pca(data)
plt.scatter(mean[0], mean[1], c='r')

# Displaying basis vectors
ax = plt.axes()
ax.arrow(mean[0], mean[1], U[0,0]*np.sqrt(s[0]), U[1,0]*np.sqrt(s[0]), head_width=0.05, head_length=0.1, fc='k', ec='k')
ax.arrow(mean[0], mean[1], U[0,1]*np.sqrt(s[1]), U[1,1]*np.sqrt(s[1]), head_width=0.05, head_length=0.1, fc='k', ec='k')

dataRot = np.dot(data,U.T)
print np.dot(U.T,U)
print np.cov(dataRot.T)
print np.dot(dataRot.T,dataRot)

plt.figure(2)
plt.scatter(dataRot[:,0], dataRot[:,1])

# Checking whitening pca
#datawhite = pca.proj_pca_white( data, U, s )
#plt.figure(3)
#plt.scatter(datawhite[:,0], datawhite[:,1])

# Checking zca 
#datazca = pca.zca_white( data, U, s )
#plt.figure(4)
#plt.scatter(datazca[:,0], datazca[:,1])

plt.show()
