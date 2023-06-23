import pickle
import pylab
import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
np.random.seed(0)


class ClassDataset:

    def __init__(self, n=1000):
        self.n = n
        X, Y = make_classification(n_samples=n, n_features=2,
                                   n_redundant=0, n_informative=2, flip_y=0.2)
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.int32)

    def train_test_split(self):
        train_x, test_x = np.split(self.X, [self.n * 8 // 10])
        train_y, test_y = np.split(self.Y, [self.n * 8 // 10])
        return train_x, train_y, test_x, test_y


class Mnist:

    def __init__(self, limit=100):
        """
        curl -o mnist.pkl.gz https://raw.githubusercontent.com/shwars/NeuroWorkshop/master/Data/MNIST/mnist.pkl.gz
        gzip -d mnist.pkl.gz
        """
        with open('neuro_learner/mnist.pkl', 'rb') as f:
            self.MNIST = pickle.load(f)
        self.labels = self.MNIST['Train']['Labels']
        self.features = self.MNIST['Train']['Features']

    def show_number_by_position(self, position):
        pylab.imshow(self.features[position].reshape(28, 28))
        print(self.labels[position])
        pylab.show()

    def train_test_split(self):
        return train_test_split(self.features, self.labels, train_size=0.9)
