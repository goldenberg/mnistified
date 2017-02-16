from flask import Flask
from flask import send_file
from flask.json import jsonify
from mnistified import datasets
from PIL import Image
from StringIO import StringIO
app = Flask(__name__)

mnist = datasets.MNIST()


@app.route('/status')
def status():
    return jsonify({
        'status': 'ok'
    })

@app.route('/mnist/classify', methods=('POST',))
def classify():
    # prediction =
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
