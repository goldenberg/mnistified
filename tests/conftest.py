import pytest
from mnistified.app import app as mnist_app


@pytest.fixture
def app():
    return mnist_app
