import numpy as np
import pytest
from keras.layers import Activation, Convolution2D
from mnistified.model import MNIST_DEFAULT_WEIGHTS_PATH, CNNModel
from numpy.testing import assert_array_almost_equal


@pytest.fixture
def model():
    m = CNNModel()
    m.load_weights(MNIST_DEFAULT_WEIGHTS_PATH)
    return m


@pytest.fixture
def untrained_model():
    return CNNModel()


# Test that the untrained model behaves deterministically on toy input.
# We'd expect these tests to fail if the model architecture changes.

def test_untrained_model_zeros(untrained_model):
    assert_array_almost_equal(
        untrained_model.classify(np.zeros((28, 28))),
        np.array([
            0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1
        ])
    )


def test_untrained_model_ones(untrained_model):
    assert_array_almost_equal(
        untrained_model.classify(np.ones((28, 28))),
        np.array([
            0.124937,  0.106037,  0.099581,  0.087444,  0.092687,  0.070209,
            0.112726,  0.065086,  0.110448,  0.130845
        ])
    )


# Test the trained model with the current weights. This might not make sense to
# do as unit tests in a "real" deployment of an ML model, where the weights
# might be retrained without changing the code. We'd have a separate process
# around QA'ing new weights.
def test_trained_model_zeros(model):
    assert_array_almost_equal(
        model.classify(np.zeros((28, 28))),
        np.array([
            0.0762402,  0.10257474,  0.10162982,  0.13408203,  0.08835191,
            0.14918971,  0.07202588,  0.11403066,  0.07073943,  0.0911357
        ])
    )


def test_trained_model_ones(model):
    assert_array_almost_equal(
        model.classify(np.ones((28, 28))),
        np.array([
            0.431459,  0.01586,  0.021785,  0.036961,  0.004694,  0.012518,
            0.159873,  0.005454,  0.303411,  0.007984
        ])
    )


def test_model_definition(model):
    # Just some basic tests that the model has the rough architecture we expect

    assert len(model.model.layers) == 12
    assert isinstance(model.model.layers[0], Convolution2D)
    assert isinstance(model.model.layers[1], Activation)

    # Check that the last layer is a softmax with 10 output nodes
    assert isinstance(model.model.layers[-1], Activation)
    assert model.model.layers[-1].activation.func_name == 'softmax'
    assert model.model.layers[-1].output_shape == (None, 10)
