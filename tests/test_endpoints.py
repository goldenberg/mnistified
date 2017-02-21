import pytest


@pytest.fixture
def image7():
    return open('tests/fixtures/mnist/7.jpg').read()


@pytest.fixture
def image7_png():
    return open('tests/fixtures/mnist/7.png').read()


def test_status(client):
    res = client.get('/status')
    assert res.status_code == 200
    assert res.json == {'status': 'ok'}


def test_get_image(client, image7):
    res = client.get('/mnist/image/7')
    assert res.status_code == 200
    assert res.data == image7


def test_get_missing_image(client):
    res = client.get('/mnist/image/100000000')
    assert res.status_code == 404


def test_classify(client, image7):
    res = client.post('/mnist/classify', data=image7)
    assert res.status_code == 200
    assert res.json['prediction'] == 3
    assert 0 < res.json['elapsed_time_ms'] < 1000
    assert res.json['debug'] == {
        'probabilities': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }


def test_classify_png(client, image7_png):
    res = client.post('/mnist/classify', data=image7_png)
    assert res.status_code == 200
    assert res.json['prediction'] == 3
