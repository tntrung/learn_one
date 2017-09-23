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


