
import numpy as np

def zero_mean(X):
	mean = np.mean(X,axis=0)
	E = X - mean[np.newaxis,:]
	return E,mean
	
