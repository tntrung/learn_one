# The demo of PCA algorithm

import numpy as np
import matplotlib.pyplot as plt

from ml.manifolds.unsupervised.pca import pca
from misc import cvmlPlot as dp
from misc import cvmlNormalization as zm
from misc import cvmlMatrix as mat

# Load patch data
data = np.loadtxt("data/patches_ascii.txt").transpose()

# Visualize randomly patches from data
#sel = np.random.randint(data.shape[0], size=100)
sel = range(100)
#dp.display_patches(data[sel,:])

# Normalizing zero mean 2D data
data,mean = zm.zero_mean_common(data)
plt.figure(1)
dp.display_patches(data[sel,:])

# Rotate data, notice that U (not U.T)
U,s = pca.pca(data)
Xrot = np.dot(data,U)

# Displaying the covariance matrice.
# It should be diagnal matrice.
covr = np.dot(Xrot.T,Xrot)/Xrot.shape[0]
plt.figure(2) 
plt.imshow(covr);

# Get 99% the variance of data
acc = np.cumsum(s)/ np.sum(s)
idx = np.where( acc < 0.99 )

# Project the data into subspace
Xhat   = Xrot[:,idx].reshape(data.shape[0],len(idx[0]))
Xerr   = np.zeros((Xhat.shape[0],data.shape[1]-Xhat.shape[1]))
Xhat   = np.hstack((Xhat,Xerr))
Xhat   = np.dot(Xhat,U.T);
plt.figure(3)
dp.display_patches(Xhat[sel,:])

# PCA whitening
XWhite = pca.proj_pca_white( data, U, s )
covr = np.dot(XWhite.T,XWhite)/XWhite.shape[0]
plt.figure(4)
plt.imshow(covr);

#ZCA 
Xzca = np.dot(XWhite,U.T)
plt.figure(5)
dp.display_patches(Xzca[sel,:])

#print sigma
plt.show()
