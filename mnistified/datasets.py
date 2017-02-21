import numpy as np
np.random.seed(42)
from keras.datasets import mnist

class MNIST(object):
    def __init__(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

    def get(self, idx):
        """Get the specified image from the X training data.

        Args:
            idx: between 0 and 60000

        Returns:
            28x28 numpy array

        Raise:
            IndexError if the index is out of bounds
        """
        return self.X_train[idx]
