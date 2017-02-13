.PHONY: all install test tests clean docs

all: install build

# Check that we're in a virtualenv, so we don't globally pip install a bunch of packages.
check-venv:
ifndef VIRTUAL_ENV
    $(error Must be in a python virtualenv)
endif

service: check-venv
	python run.py

install: check-venv
	pip install -r requirements.txt

tests: test

test:
	tox

clean:
	rm -rf .tox build dist src docs/build *.egg-info .eggs .cache .coverage
	find . -name '*.pyc' -delete
	find . -type d -a -name '__pycache__' | xargs rm -rf