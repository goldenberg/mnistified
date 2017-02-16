import numpy as np

from flask import Flask
from flask import request
from flask import send_file
from flask.json import jsonify
from mnistified import datasets
from mnistified.model import CNNModel
from PIL import Image
from StringIO import StringIO
import os.path

app = Flask(__name__)

mnist = datasets.MNIST()
model = CNNModel()
model.load_weights(os.path.join(os.path.dirname(__file__), '../model.hdf5'))

@app.route('/status')
def status():
    return jsonify({
        'status': 'ok'
    })

@app.route('/mnist/classify', methods=('POST',))
def classify():

    # model = CNNModel.from_hd5(os.path.join(os.path.dirname(__file__), '../model.hdf5'))
    img = Image.open(StringIO(request.data))
    img_array = np.array(img)
    prediction = model.classify(img_array)

    return jsonify({
        'prediction': 9,
        'elapsed_time_ms': 3,
        'debug': {
            'foo': 'bar'
        }
    })

@app.route('/mnist/image/<idx>')
def get_image(idx):
    img = Image.fromarray(mnist.get(int(idx)))
    img_io = StringIO()
    img.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')
