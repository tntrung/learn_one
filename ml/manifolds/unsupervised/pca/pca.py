# This modules develops the Principal Component Analysis (PCA) algorithm
# and its variations

import numpy as np

# The objective function of PCA as follows:
#
#   argmax w^T*S*w
#   s.t.   w^T*w = I
#
#   where S is covariance matrix, identity matrix I.

def pca( X ):
	
	# computing mean vector
	mean = np.mean(X,axis=0)

	# zero mean the data
	E = X - mean[np.newaxis,:]

	# computing pca basis
	sigma = np.cov(E.T)
	U, s, V = np.linalg.svd(sigma, full_matrices=True)

	return (U, s, mean)
