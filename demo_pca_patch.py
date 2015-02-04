# The demo of PCA algorithm

import numpy as np
import matplotlib.pyplot as plt

from ml.manifolds.unsupervised.pca import pca
from misc import display_patches as dp, zero_mean as zm

# Load patch data
data = np.loadtxt("data/patches_ascii.txt").transpose()

# Visualize randomly patches from data
#sel = np.random.randint(data.shape[0], size=100)
sel = range(100)
#dp.display_patches(data[sel,:])

# Normalizing zero mean 2D data
data = zm.zero_mean(data)
#dp.display_patches(data[sel,:])

# Rotate PCA
U,s = pca.pca(data)

# Rotate data, notice that U (not U.T)
Xrot = np.dot(data,U)

covr = np.dot(Xrot.T,Xrot)/Xrot.shape[0]

#print covr

plt.imshow(covr);

#print sigma
plt.show()
