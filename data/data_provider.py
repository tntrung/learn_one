import cifar10

class DataProvider(object):

	# Constructor
	def __init__(self, db_name = 'None', db_path = 'None'):
		"""
		Init DataProvider with db_name and db_path provided

		Input
		---
			db_name: the database name, default (mnist)
			db_path: the database path, default ('./db_name/')		
		"""
		db_list = ['mnist', 'cifar10']
		self.db_name = db_name
		self.db_path = db_path
		
		# check whether db_name is supported
		if any(self.db_name in s for s in db_list):
			self.db_name = db_name
			if self.db_path == "None":
				self.db_path = './data/' + self.db_name
		else:
			print "Your database ", db_name, " is not suppored!"
		

	def load_data(self, reshape = True):
		"""
		Load datasets
		"""

		print "Loading", self.db_name , "..."

		# Load cifar dataset:
		# original link: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
		# download manually and uncompress it in correct path
		if self.db_name == 'cifar10':
			Xtrain, Ytrain, Xtest, Ytest = cifar10.load_CIFAR10(self.db_path)
			if reshape == True:
				Xtrain = Xtrain.reshape(Xtrain.shape[0], 32 * 32 * 3)
				Xtest  = Xtest.reshape(Xtest.shape[0], 32 * 32 * 3)
			return Xtrain, Ytrain, Xtest, Ytest
			
		
