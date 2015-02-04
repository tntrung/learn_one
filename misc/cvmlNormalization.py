
import numpy as np

def zero_mean_common(X):
	mean = np.mean(X,axis=0)
	E = X - mean[np.newaxis,:]
	return E,mean


def zero_mean(X):
	mean = np.mean(X,axis=1)
	E = X - mean[:,np.newaxis]
	return E
	
