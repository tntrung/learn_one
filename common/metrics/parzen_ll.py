#python 
import numpy as np
import gc


def log_mean_exp(a):
    """
    .. todo::
        WRITEME
    """
    max_ = a.max(1) # Get max values of rows
    return max_ + np.log(np.exp(a - max_[:,None]).mean(1))


def gaussian_parzen(X, samples, sigma):
	"""
	Compute gaussian parzen of data from X to samples, given sigma of gaussian.

	Parameters
    	-----------
	X: numpy matrix
		N x d, N: number of training samples, d: image dimension (1D)
	samples: numpy matrix
		M x d, N: number of generated samples, d: image dimension (1D)
	sigma: scalar 
		the standard variation of gaussian

    	Returns
	-----------
	
	"""
	# Parzen: the distance of each element of X to all elements of "samples"
        # Expected output: N x M

	# X[:, None, :]: reshape to [N,1,d]
        # samples[None, :, :]: reshape to [1,M,d]
        # X[:, None, :] - samples[None, :, :]: broadcasting to [N,M,d]
	A = (X[:, None, :] - samples[None, :, :]) / sigma
	
	return log_mean_exp(-0.5 * (A**2).sum(2)) - samples.shape[1] * np.log(sigma * np.sqrt(np.pi * 2))		
	

def get_ll(X, samples, sigma, batch_size=10):
	"""

	"""
	inds = range(X.shape[0])
	nb_data    = float(len(inds))
	nb_batches = int(np.ceil(nb_data / batch_size))
	
	nlls = []
	for i in range(nb_batches):
		data = X[inds[i::nb_batches]]
		nll  = gaussian_parzen(data, samples, sigma)
		nlls.extend(nll)

	return np.array(nlls)

def cross_validate_sigma(samples, val_data, sigmas, batch_size):
	"""
	Estimate sigma of parzen window via validation data.
	"""
	lls = []
	for sigma in sigmas:
		tmp = get_ll(val_data, samples, sigma, batch_size)	
		lls.append(np.asarray(tmp).mean())
		print "parzen_ll::Sigma = %0.5f, Mean log-Likelihood = %0.5f" % (sigma,np.asarray(tmp).mean())
		del tmp
		gc.collect()

	ind = np.argmax(lls)
	return sigmas[ind]

def easy_evaluate(samples, X_val, X_test, sigma_start = -1, sigma_end = 1, cross_val  = 10, batch_size = 10):
	"""
	Evaluate easily with the input of generated samples, validation and set sets.
	"""
	# sigma range
	sigma_range = np.logspace(sigma_start, sigma_end, cross_val)

	# cross validate for best sigma
	sigma = cross_validate_sigma(samples, X_val, sigma_range, batch_size)

	# test data
	return get_ll(X_test, samples, sigma, batch_size)
