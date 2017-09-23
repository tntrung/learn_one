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
			
			
	def cross_validation(self, X, Y, karr, verbose = 0):
			
		Xdev = X[:1000,:] # take first 1000 images for dev set
		Ydev = Y[:1000] # take first 1000 images for dev set
		Xtr  = X[1000:,:] # remain images as training set
		Ytr  = Y[1000:] # remain images as training set

		val_accs = []

		for k in karr:
			print 'Cross validating k = %d' %(k)
			# train model
			self.train(Xtr, Ytr, verbose)
			# predict labels
			Ypred = self.predict(Xdev, k, verbose)
			acc = np.mean(Ypred == Ydev)
			val_accs.append((k, acc))
		if (verbose == 1):
			print 'Cross validation accuracy:'	
			print val_accs
		return val_accs


	def easy(self, Xtr, Ytr, Xt, Yt, karr, cross_val = 0, verbose = 0):
		"""
		Use a single function with knn
		
		Input
		---
			Xtr, Ytr, Xt, Yt: image and label of train and test resepectively X: (N x d), Y (N x 1)
			karr: a numpy array, e.g, [1] or [1, 5, 4, 7, 6], 
			cross_val: if want to use cross validation, 0: karr[0] is used for training.
			verbose - 1: print detail information, otherwise 0
		"""
		acc = []
		if cross_val == 0:
			self.train(Xtr, Ytr, verbose)
			Yt_hat = self.predict(Xt, karr[0], verbose)
			acc = np.mean(Yt == Yt_hat)
			if (verbose == 1):
				print 'Accuracy: %f' % (acc)
		else:
			val_accs = self.cross_validation(Xtr, Ytr, karr, verbose)
			best_k = [x for x,_ in sorted(val_accs, reverse=True, key=lambda pair: pair[1])]
			if (verbose == 1):
				print 'Best accuracy according the number of nearest neighbor:'
				print best_k
			Yt_hat = self.predict(Xt, best_k[0], verbose)
			acc = np.mean(Yt == Yt_hat)
			if (verbose == 1):
				print 'Accuracy: %f with k = %d' % (acc, best_k[0])
		return acc
