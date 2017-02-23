import os.path

import pytest


def get_image_bytes(fname):
    with open(os.path.join('tests/fixtures/', fname)) as f:
        return f.read()


def test_status(client):
    res = client.get('/status')
    assert res.status_code == 200
    assert res.json['status'] == 'ok'

    # Check that the first layer is a convolution
    assert res.json['model']['layers'][0]['class_name'] == 'Convolution2D'


@pytest.mark.parametrize('idx', [
    7,
    42,
])
def test_get_images(idx, client):
    res = client.get('/mnist/image/{}'.format(idx))
    assert res.status_code == 200
    assert res.data == get_image_bytes('mnist/{}.jpg'.format(idx))


def test_get_missing_image(client):
    res = client.get('/mnist/image/100000000')
    assert res.status_code == 404


@pytest.mark.parametrize('fname,expected_label', [
    ('mnist/7.jpg', 9),
    ('mnist/7.png', 9),
    ('mnist/7.tiff', 9),
    ('mnist/7.gif', 9),
    ('mnist/42.jpg', 4),
    ('mnist/42.png', 4),
    ('mnist/42.tiff', 4),
    ('mnist/42.gif', 4),
])
def test_classify(client, fname, expected_label):
    res = client.post('/mnist/classify', data=get_image_bytes(fname))
    assert res.status_code == 200
    assert res.json['prediction'] == expected_label
    assert 0 < res.json['elapsed_time_ms'] < 1000

    expected_probabilities = [0.0] * 10
    expected_probabilities[expected_label] = 1.0
    assert res.json['debug'] == {
        'probabilities': expected_probabilities
    }


@pytest.mark.parametrize('fname', [
    'bad_images/100x100.jpg',
    'bad_images/100x28.jpg',
    'bad_images/empty.jpg',
    'bad_images/empty.png'
])
def test_bad_images(client, fname):
    """Various badly formed images, wrong size, empty files, etc."""
    res = client.post('/mnist/classify', data=get_image_bytes(fname))
    assert res.status_code == 400
