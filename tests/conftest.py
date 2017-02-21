from mnistified.app import app as mnist_app
import pytest

# TODO


@pytest.fixture
def app():
    return mnist_app
    # app = create_app()
    # return app
