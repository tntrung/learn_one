# This modules develops the Principal Component Analysis (PCA) algorithm
# and its variations

import numpy as np

# The objective function of PCA as follows:
#
#   argmax w^T*S*w or argmax |transpose(Xw)*Xw|^2
#   s.t.   w^T*w = I
#
#   where S is covariance matrix, identity matrix I.
#   assumed X is normalized zero-mean.

# === Traditional PCA ==================================================
def pca( X ):
	
	# computing mean vector
	mean = np.mean(X,axis=0)

	# zero mean the data
	E = X - mean[np.newaxis,:]

	# computing pca basis
	sigma = np.cov(E.T)
	U, s, V = np.linalg.svd(sigma, full_matrices=True)

	return (U, s, mean)

# === PCA keeps k eigen vectors ========================================
def pca_k( X , k ):
	
	U,s,mean = pca(X)
	
	return (U[0:k,:], s[0:k], mean)


# === PCA keeps p% variance retained ===================================
def pca_p( X , p ):
	
	U,s,mean = pca(X)

	acc = np.cumsum(s)/ np.sum(s)

	idx = np.where( acc < p )

	return (U[:,idx], s[idx], mean)


# === PCA whitening ====================================================
def proj_pca_white( X, U, s ):
	
	XRot = np.dot(X,U.T)

	print np.tile(s, (XRot.shape[0], 1))

	Xwhite = XRot / np.sqrt(np.tile(s, (XRot.shape[0], 1)))

	return Xwhite

# === ZCA Whitening ====================================================
def zca_white( X, U, s ):

	# avoiding if eigen values are too smalle (~ 0.0)
	s = s + 10**(-5)

	XWhite = proj_pca_white(X, U, s)

	Xzca = np.dot(XWhite,U.T);	

	return Xzca
	
