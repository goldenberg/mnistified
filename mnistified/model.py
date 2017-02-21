import logging

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.datasets import mnist
from keras.layers import (Activation, Convolution2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential
from keras.utils import np_utils

np.random.seed(42)  # for reproducibility


# Use a global Tensorflow graph, so that it is reused in the request context.
# See https://github.com/fchollet/keras/issues/2397#issuecomment-254919212
graph = tf.get_default_graph()

log = logging.getLogger(__file__)


# input image dimensions
MNIST_IMG_ROWS = 28
MNIST_IMG_COLS = 28
MNIST_NB_CLASSES = 10


class CNNModel(object):
    """Convolutional Neural Network model for classifying MNIST images.

    Copied from the examples distributed with Keras:
    https://raw.githubusercontent.com/fchollet/keras/master/examples/mnist_cnn.py
    """

    def __init__(self):
        super(CNNModel, self).__init__()
        self.model = self._define_model()
        # TODO time it?

        log.debug('Compiling TensorFlow model')
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adadelta',
                           metrics=['accuracy'])
        log.debug('Compiling TensorFlow model')

    def _define_model(self):
        """Define the convolutional model for classifying 28 x 28 digits.

        Does NOT compile the model since this is computationally expensive,
        so is done explicitly above.
        """
        # number of convolutional filters to use
        nb_filters = 32
        # size of pooling area for max pooling
        pool_size = (2, 2)
        # convolution kernel size
        kernel_size = (3, 3)

        if K.image_dim_ordering() == 'th':
            input_shape = (1, MNIST_IMG_ROWS, MNIST_IMG_COLS)
        else:
            input_shape = (MNIST_IMG_ROWS, MNIST_IMG_COLS, 1)

        model = Sequential()
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                border_mode='valid',
                                input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(MNIST_NB_CLASSES))
        model.add(Activation('softmax'))

        return model

    def train(self, batch_size=128, num_epochs=12):
        """Train the model using specified batch size and epochs."""

        # Load the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        if K.image_dim_ordering() == 'th':
            X_train = X_train.reshape(X_train.shape[0], 1, MNIST_IMG_ROWS, MNIST_IMG_COLS)
            X_test = X_test.reshape(X_test.shape[0], 1, MNIST_IMG_ROWS, MNIST_IMG_COLS)
        else:
            X_train = X_train.reshape(X_train.shape[0], MNIST_IMG_ROWS, MNIST_IMG_COLS, 1)
            X_test = X_test.reshape(X_test.shape[0], MNIST_IMG_ROWS, MNIST_IMG_COLS, 1)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        log.debug('X_train shape:', X_train.shape)
        log.debug(X_train.shape[0], 'train samples')
        log.debug(X_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, MNIST_NB_CLASSES)
        Y_test = np_utils.to_categorical(y_test, MNIST_NB_CLASSES)

        self.model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=num_epochs,
                       verbose=1, validation_data=(X_test, Y_test))

    def serialize(self, path):
        """Save the model's weights to the specified file path."""
        self.model.save_weights(path)

    @classmethod
    def from_hd5_weights(cls, path):
        m = CNNModel()
        m.load_weights(path)

    def load_weights(self, path):
        """Load serialized trained weights for this model.

        Args:
            path - path to an HDF5 file of serialized weights.
        """
        self.model.load_weights(path)

    def classify(self, img):
        """Classify an input image of a handwritten digit from the MNIST dataset.

        Args:
            img - 2D 28 x 28 pixel image, as a numpy array

        Returns:
            Integer in [0, 9]

        Raises:
            ValueError if the input is the wrong dimensions
        """
        if img.ndim != 2:
            raise ValueError("Expected a 2D, 28 x 28 image.")
        if img.shape != (MNIST_IMG_ROWS, MNIST_IMG_COLS):
            raise ValueError("Expected a 2D image. Got {}".format(img.shape))

        # Convert the input into a 4D tensor, which the model expects.
        # The four dimensions are: image index, channel, x, y
        # In a "real" implementation, we'd likely want to bulk process images on the
        # first dimension in order to take advantage of the GPUs parallelism.
        model_input = np.array([[img]])

        # Use the global TF graph defined above.
        global graph
        with graph.as_default():
            return self.model.predict(model_input)
