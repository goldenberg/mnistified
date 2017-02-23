import datetime
from io import BytesIO

import numpy as np
from flask import Flask, request, send_file
from flask.json import jsonify
from mnistified import datasets
from mnistified.model import MNIST_DEFAULT_WEIGHTS_PATH, CNNModel
from PIL import Image
from webargs import fields
from webargs.flaskparser import use_kwargs
from werkzeug import exceptions

app = Flask(__name__)

# Load the dataset for accessing the images
mnist = datasets.MNIST()

# Initialize the model and load stored weights
model = CNNModel()

# XXX: In a production implementation, I'd make this a configurable variable
# managed by our configuration management system.
model.load_weights(MNIST_DEFAULT_WEIGHTS_PATH)


@app.route('/status')
def status():
    """Healthcheck endpoint to use for monitoring the status of the service.

    In a production deployment, this might be used both for monitoring and alerting
    and also as a load balancer health check.
    """
    return jsonify({
        'status': 'ok',
        'model': {
            'layers': model.model.get_config()
        },
    })


@app.route('/mnist/classify', methods=('POST',))
@app.route('/mnist/classify/<idx>', methods=('GET',))
@use_kwargs({'idx': fields.Integer(location='view_args', minimum=0, required=False)})
def classify(idx=None):
    """Classify an image passed in the POST request body.

    Request body:
        Any 28x28 grey-scale image parse-able by the Python Image Library, including
        JPEG and PNG files.

    Status codes:
        200: successful classification
        400: Improperly formatted or sized image
    """
    start_time = datetime.datetime.now()

    if request.method == 'POST':
        try:
            img = Image.open(BytesIO(request.data))
            img_array = np.array(img)
        except (ValueError, IOError) as e:
            raise exceptions.BadRequest(e)
    elif request.method == 'GET':
        try:
            img_array = mnist.get_test_image(idx)
        except IndexError as e:
            raise exceptions.NotFound(e)

    # ValueErrors are thrown for improperly sized input
    try:
        prediction = model.classify(img_array)
    except ValueError as e:
        raise exceptions.BadRequest(e)

    max_class = np.argmax(prediction)
    elapsed_time = datetime.datetime.now() - start_time

    return jsonify({
        'prediction': max_class,
        'elapsed_time_ms': elapsed_time.total_seconds() * 1000,
        'debug': {
            'probabilities': prediction.tolist()
        }
    })


@app.route('/mnist/image/<idx>')
@use_kwargs({'idx': fields.Integer(location='view_args', minimum=0, required=True)})
def get_image(idx):
    """Access individual images in the MNIST dataset.

    The dataset is deterministcally shuffled, so indices will be consistent.

    Path parameters:
        idx - integer index in the shuffled dataset

    Status Codes:
        200: JPEG encoded image response
        400: BadRequest if the index can't be parsed
        404: File Not Found because of invalid index
    """
    try:
        img_array = mnist.get_test_image(idx)
    except IndexError as e:
        raise exceptions.NotFound(e)

    img = Image.fromarray(img_array)
    img_io = BytesIO()
    img.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')


@app.route('/mnist/label/<idx>')
@use_kwargs({'idx': fields.Integer(location='view_args', minimum=0, required=True)})
def get_image_label(idx):
    """Access the label of individual image in the MNIST dataset.

    The dataset is deterministcally shuffled, so indices will be consistent.

    Path parameters:
        idx - integer index in the shuffled dataset

    Status Codes:
        200: JPEG encoded image response
        400: BadRequest if the index can't be parsed
        404: File Not Found because of invalid index
    """
    try:
        img_label = mnist.get_test_label(idx)
    except IndexError as e:
        raise exceptions.NotFound(e)

    return jsonify({
        'label': img_label
    })
