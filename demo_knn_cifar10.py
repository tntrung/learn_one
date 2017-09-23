import numpy as np
from data import data_provider as dp
from ml.classification.knn import knn

# load cifar10
data = dp.DataProvider('cifar10')
[Xtr, Ytr, Xt, Yt] = data.load_data()

# classify images
classifier = knn.KNearestNeighbor()
# train
classifier.train(Xtr, Ytr, verbose = 1)
# predict with k = 5
Yt_hat = classifier.predict(Xt, k = 5, verbose = 1)

# accuracy
acc = np.mean(Yt == Yt_hat)
print 'Accuracy: %f' % (acc)

# uncomment if you want to use

# cross validation 
# acc_list = classifier.cross_validation(Xtr, Ytr, [1,2,3,4,5,6,7,8,9], verbose = 1)
# print "Acc list:"
# print acc_list

# easy use knn
# acc = classifier.easy(Xtr, Ytr, Xt, Yt, karr = [1, 3, 5], cross_val = 1, verbose = 1)
# print 'Accuracy: %f' % (acc)



