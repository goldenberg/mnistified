MNISTified
==========

MNISTified is an REST service that classifies images of handwritten digits from
the MNIST dataset.


# Getting started

The Makefile contains targets for model training, starting the service, and
running the automated tests. It assumes that it will be executed within a
virtual environment.

## Service
Setup and install the service
```
./mnistified $ virtualenv venv
./mnistified $ make
pip install -r requirements.txt
....
```

Then, to start to the service, run `make service`.
```
./mnistified $ make service
PYTHONPATH=./ FLASK_APP=mnistified/app.py flask run
Using TensorFlow backend
 * Serving Flask app "mnistified.app"
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

Confirm that the service is up, and the model loaded, by querying `/status`:
```
$ curl http://127.0.0.1:5000/status
{
    "model": {
        "layers": {
            ...
        }
    },
    "status": "ok"
}
```

## Classification

Classify an input image:

TODO: make sure paths are right
```
curl --data-binary @tests/fixtures/mnist/7.png -X POST "http://127.0.0.1:5000/mnist/classify" --header "Content-Type:image/png"
{
  "debug": {
    "probabilities": [
      0.0,
      0.0,
      0.0,
      1.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  },
  "elapsed_time_ms": 41.961,
  "prediction": 3
}
```

## Training

This git repository includes both the code to train the model, and a serialized
set of trained model weights. The model can be retrained using the default
parameters with the `make train` target, or for just one epoch using `make
fast-train`.

Or, the `train.py` script can be used:

```PYTHONPATH=./ python mnistified/train.py --help
Using TensorFlow backend.
usage: train.py [-h] [--weights WEIGHTS] [--num-epochs NUM_EPOCHS]
                [--batch-size BATCH_SIZE]

Train an MNIST model and serialize the model weights.

optional arguments:
  -h, --help            show this help message and exit
  --weights WEIGHTS     HDF5 output file for the weights
  --num-epochs NUM_EPOCHS
                        Number of epochs to train for.
  --batch-size BATCH_SIZE
                        Mini batch size for training.
```

Each epoch takes several minutes on a consumer laptop (e.g. Macbook Pro) without
a high end GPU.

# Testing

Unit and integration tests are run via `tox` which can also be run with a make
target. `tox` will run the tests in a dedicated virtual environment, and also
compute a test coverage report.

# TODO change coverage report
```
$ make test
tox
py27 create: /Users/bgoldenberg/code/local/mnistified/.tox/py27
py27 installdeps: -rrequirements.txt, -rrequirements_test.txt
...
tests/test_endpoints.py::test_status PASSED
tests/test_endpoints.py::test_get_image PASSED
tests/test_endpoints.py::test_get_missing_image PASSED
tests/test_endpoints.py::test_classify PASSED
tests/test_endpoints.py::test_classify_png PASSED

---------- coverage: platform darwin, python 2.7.13-final-0 ----------
Name                      Stmts   Miss  Cover
---------------------------------------------
mnistified/__init__.py        0      0   100%
mnistified/app.py            37      2    95%
mnistified/datasets.py        8      0   100%
mnistified/mnist_cnn.py      51     51     0%
mnistified/model.py          74     22    70%
mnistified/train.py          13     13     0%
---------------------------------------------
TOTAL                       183     88    52%


======================== 5 passed in 0.10 seconds =========================
_________________________________ summary _________________________________
  py27: commands succeeded
  congratulations :)
```


# Next steps

* The raw binary weights file is checked into Git for simplicity. They should be
managed using [Git LFS](https://git-lfs.github.com/) or another system for
distributing large binary files. It might also make sense to track the testing
images using Git LFS.

* For a "production deployment", we would run multiple worker processes, e.g.
managed by [uwsgi](https://uwsgi-docs.readthedocs.io/en/latest/).

* Depending on the environment, it may be better to install Tensorflow and
Keras as system packages or from source instead of via `pip` so they take
advantage of faster GPU and CPU instructions.

* The biggest opportunity for speed optimization and scalability is to build a
bulk endpoint that takes multiple images and classifies all of them. A GPU
should be able to parallelize model execution extremely well.

* Depending on the product motivation, we'd likely want to be able to load
multiple versions of the MNIST models: either with different sets of weights or
different architectures or both.
