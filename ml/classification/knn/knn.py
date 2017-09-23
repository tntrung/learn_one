import numpy as np

class KNearestNeighbor(object):

	def __init__(self):
		"""
		Done.
		"""

	def train(self, X, Y, verbose = 0):
		""" X is N x D where each row is an example. Y is 1-dimension of size N """
    		# the nearest neighbor classifier simply remembers all the training data
		if (verbose == 1):
				print 'Training K nearest neighbor classifier ... Done!'
		self.Xtr = X
		self.Ytr = Y
		

	def predict(self, X, k = 1, verbose = 0):
		""" X is N x D where each row is an example we wish to predict label for """

		if (verbose == 1):
				print 'Start predicting labels ... '
		nb_test = X.shape[0]

		# lets make sure that the output type matches the input type
		Ypred = np.zeros((nb_test,1), dtype = self.Ytr.dtype)

		# loop over all test samples
		for i in range(nb_test):
			########################################################################
		        # TODO:                                                                 #
		        # Use the distance matrix to find the k nearest neighbors of the ith    #
		        # training point, and find the labels of these      			#
		        # neighbors. Store these labels in closest_y.                           #
		        # Hint: Look up the function numpy.argsort.                             #
		        #########################################################################
			# find the nearest training image to the i'th test image
      			# using the L1 distance (sum of absolute value differences)				
			dists = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
			indices  = dists.argsort()[:k] # get the index with k smallest distances
			closets  = self.Ytr[indices]
			if (verbose == 1):
				print k, 'nearest labels:', closets, ', dists:', dists[indices]
                        label    = np.argmax(np.bincount(closets))
			Ypred[i] = label  # predict the label of the nearest example
			if (verbose == 1):
				print 'Final label of %d-th image: %d' %(i, Ypred[i])	
		if (verbose == 1):
				print 'Done!'		
		return Ypred
			
			
		

