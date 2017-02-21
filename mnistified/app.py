import datetime
import os.path
from io import BytesIO

import numpy as np
from flask import Flask, request, send_file
from flask.json import jsonify
from mnistified import datasets
from mnistified.model import CNNModel
from PIL import Image
from werkzeug import exceptions

app = Flask(__name__)

# Load the dataset for accessing the images
mnist = datasets.MNIST()

# Initialize the model and load stored weights
model = CNNModel()

# TODO: config this?
model.load_weights(os.path.join(os.path.dirname(__file__), '../model.hdf5'))


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
def classify():
    """Classify an image passed in the POST request body.

    Request body:
        Any 28x28 grey-scale image parse-able by the Python Image Library, including
        JPEG and PNG files.

    Status codes:
        200: successful classification
        400: Improperly formatted or sized image
    """
    start_time = datetime.datetime.now()

    try:
        img = Image.open(BytesIO(request.data))
        img_array = np.array(img)
        prediction = model.classify(img_array)
    except ValueError as e:
        raise exceptions.BadRequest(e)

    max_class = np.argmax(prediction)
    elapsed_time = datetime.datetime.now() - start_time

    return jsonify({
        'prediction': max_class,
        'elapsed_time_ms': elapsed_time.total_seconds() * 1000,
        'debug': {
            'probabilities': prediction.tolist()[0]
        }
    })


@app.route('/mnist/image/<idx>')
def get_image(idx):
    """Access individual images in the MNIST dataset.

    The dataset is deterministcally shuffled, so indices will be consistent.

    Path parameters:
        idx - integer index in the shuffled dataset

    Status Codes:
        200: JPEG encoded image response
        404: File Not Found because of invalid index
    """
    # TODO: error check idx format
    try:
        img_array = mnist.get(int(idx))
    except IndexError as e:
        raise exceptions.NotFound(e)

    img = Image.fromarray(img_array)
    img_io = BytesIO()
    img.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')
