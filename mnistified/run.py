from flask import Flask
from flask.json import jsonify
app = Flask(__name__)


@app.route('/status')
def status():
    return jsonify({
        'status': 'ok'
    })