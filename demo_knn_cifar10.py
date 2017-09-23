from data import data_provider as dp
from ml.classification.knn import knn

# load cifar10
data = dp.DataProvider('cifar10')
[Xtr, Ytr, Xt, Yt] = data.load_data()

# classify images
classifier = knn.KNearestNeighbor()
classifier.train(Xtr, Ytr, 1)
Yt_hat = classifier.predict(Xt, 5, 1)

# accuracy
acc = np.mean(Yval_predict == Yval)
print 'Accuracy: %f' % (acc)


