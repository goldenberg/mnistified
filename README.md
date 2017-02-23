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

## Dataset
The test examples from the MNIST dataset are available at
`http://127.0.0.1:5000/mnist/image/<idx>`. For example, you should be able to
open this URL in a browser and view a JPEG of a handwritten image of a 4
`http://127.0.0.1:5000/mnist/image/42`.

And view the label
```
$ curl "http://127.0.0.1:5000/mnist/label/42"
{
  "label": 4
}
```
## Classification

Images can be classified with two endpoints, either by POSTing a 28x28 image file to
`http://127.0.0.1:5000/mnist/classify` or by specifying an image index:

We output the prediction, the elapsed wall time to evaluate the model, as well
as the probabilities outputted by the model for debugging. Some sample images
are available at `tests/fixtures/mnist` in various file formats.

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

Or by specifying an image index:
```
$ curl "http://127.0.0.1:5000/mnist/classify/42"
{
  "debug": {
    "probabilities": [
      0.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  },
  "elapsed_time_ms": 21.950999999999997,
  "prediction": 4
}

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

Unit and integration tests are run via `tox` which can also be run via `make
test`. `tox` will run the tests in a dedicated virtual environment, and also
compute a test coverage report.

# TODO change coverage report
```
$ make test
tox
...
collected 21 items

tests/test_endpoints.py::test_status PASSED
tests/test_endpoints.py::test_get_images[7] PASSED
tests/test_endpoints.py::test_get_images[42] PASSED
...

---------- coverage: platform darwin, python 2.7.13-final-0 ----------
Name                      Stmts   Miss  Cover
---------------------------------------------
mnistified/__init__.py        0      0   100%
mnistified/app.py            57     14    75%
mnistified/datasets.py       10      1    90%
mnistified/mnist_cnn.py      51     51     0%
mnistified/model.py          76     21    72%
mnistified/train.py          13     13     0%
---------------------------------------------
TOTAL                       207    100    52%


======================================================== 21 passed in 2.40 seconds ========================================================
_________________________________________________________________ summary _________________________________________________________________
  py27: commands succeeded
  congratulations :)
```


# Next steps

* The raw binary weights file is checked into Git for simplicity. They should be
managed using [Git LFS](https://git-lfs.github.com/) or another system for
distributing large binary files. It might also make sense to track the testing
images using Git LFS.

* For a "production deployment", we would run multiple worker processes, e.g.
managed by [uwsgi](https://uwsgi-docs.readthedocs.io/en/latest/), likely proxied
through nginx or Apache.

* Depending on the environment, it may be better to install Tensorflow and
Keras as system packages or from source instead of via `pip` so they take
advantage of faster GPU and CPU instructions.

* The biggest opportunity for speed optimization and scalability is to build a
bulk endpoint that takes multiple images and classifies all of them. A GPU
should be able to parallelize model execution extremely well.

* Depending on the product motivation, we'd likely want to be able to load
multiple versions of the MNIST models: either with different sets of weights or
different architectures or both.

* The current implementation verifies that the model is working as expected by
executing unit tests that classify known images. If we expect the weights to be
regularly updated, unit tests are probably not the right implementation for QA.
Instead, we'd implement a QA process that verifies the model's output on a
representative set of data, and compares it to known baselines.

* For a pure TensorFlow environment, [Tensorflow
Serving](https://tensorflow.github.io/serving/) might be a good choice to manage
the retraining lifecycle.

* The API might be a bit cleaner by swapping the order of the path components.
If we changed the format from `/mnist/image/<idx>`, `/mnist/classify/<idx>` etc.
to `/mnist/<idx>/image`, `/mnist/<idx>/classify`, it would emphasize the
resource being specified, and then the actions to take on that resource. But I
wanted to make it consistent with the URL specified in the instructions.

* The argument parsing and validation is a bit of a mess, but should return reasonable status codes. In the past I've used [webargs]
