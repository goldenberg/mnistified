.PHONY: all install test tests clean docs

all: install

# Check that we're in a virtualenv, so we don't globally pip install a bunch of packages.
check-venv:
ifndef VIRTUAL_ENV
    $(error Must be in a python virtualenv)
endif

# A "real" deployment would be deployed with multiple worker processes
# using uwsgi.
service: check-venv
	PYTHONPATH=./ FLASK_APP=mnistified/app.py flask run

# FLASK_DEBUG=1 allows interactive debugging and better tracebacks
debug:
	PYTHONPATH=./ FLASK_APP=mnistified/app.py FLASK_DEBUG=1 flask run

install: check-venv
	pip install -r requirements.txt

tests: test

test:
	tox

train: check-venv
	PYTHONPATH=./ python mnistified/train.py --weights=data/weights.hdf5

# Train for a single epoch
fast-train: check-venv
	PYTHONPATH=./ python mnistified/train.py --num-epochs=1 --weights=data/weights.hdf5

clean:
	rm -rf .tox build dist src docs/build *.egg-info .eggs .cache .coverage
	find . -name '*.pyc' -delete
	find . -type d -a -name '__pycache__' | xargs rm -rf