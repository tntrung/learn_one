import numpy as np

class KNearestNeighbor(object):

	def __init__(self):
		"""
		Done.
		"""

	def train(X, Y):
		self.Xtr = X
		self.Ytr = Y


	def predict(X):
		"""
		Predict the labels of X.
		"""		
		nb_test = X.shape[0]
		Ypred = np.zeros(nb_test,1, dtype= self.Ytr.dtype)
		for i in range(nb_test):
			distances = np.sum(np.abs(self.Xtr - X[:,i]), axis = 1)
			min_index = np.argmin(distances)
			Ypred[i]  = self.Ytr[min_index]
		return Ypred
			
			
		

