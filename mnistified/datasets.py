import numpy as np
np.random.seed(42)
from keras.datasets import mnist

class MNIST(object):
    def __init__(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

    def get(self, idx):
        return self.X_train[idx]
