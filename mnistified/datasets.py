import numpy as np
from keras.datasets import mnist

np.random.seed(42)


class MNIST(object):
    """Wrapper for the MNIST data distributed with Keras.

    Data is randomly preshuffled between training and split, but a consistent
    seed is used for reproducibility.
    """

    def __init__(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

    def get_test_image(self, idx):
        """Get the specified image from the X testing data.

        Args:
            idx - integer

        Returns:
            28x28 numpy array

        Raise:
            IndexError if the index is out of bounds
        """
        return self.X_test[idx]

    def get_test_label(self, idx):
        """Get the specified test example label.

        Args:
            idx - integer

        Returns:
            [0 -9]

        Raise:
            IndexError if the index is out of bounds
        """
        return int(self.y_test[idx])
