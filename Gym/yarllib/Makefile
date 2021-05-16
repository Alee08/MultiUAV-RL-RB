.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test clean-docs ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-docs:  ## remove MkDocs products.
	mkdocs build --clean
	rm -fr site/


clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .mypy_cache
	rm -fr coverage.xml

lint-all: black isort lint static bandit safety vulture pylint ## run all linters

lint: ## check style with flake8
	flake8 src/yarllib tests scripts

static: ## static type checking with mypy
	mypy src/yarllib tests scripts

isort: ## sort import statements with isort
	isort src/yarllib tests scripts

isort-check: ## check import statements order with isort
	isort --check-only src/yarllib tests scripts

black: ## apply black formatting
	black src/yarllib tests scripts

black-check: ## check black formatting
	black --check --verbose src/yarllib tests scripts

bandit: ## run bandit
	bandit src/yarllib tests scripts

safety: ## run safety
	safety

pylint: ## run pylint
	pylint src/yarllib tests scripts

vulture: ## run vulture
	vulture src/yarllib scripts/whitelist.py

test: ## run tests quickly with the default Python
	pytest tests --doctest-modules \
        src/yarllib tests/ \
        --cov=src/yarllib \
        --cov-report=xml \
        --cov-report=html \
        --cov-report=term

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	coverage run --source src/yarllib -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate MkDocs HTML documentation, including API docs
	mkdocs build --clean
	$(BROWSER) site/index.html

servedocs: docs ## compile the docs watching for changes
	mkdocs build --clean
	python -c 'print("###### Starting local server. Press Control+C to stop server ######")'
	mkdocs serve

release: dist ## package and upload a release
	twine upload dist/*

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	python setup.py install

develop: clean ## install the package in development mode
	pip install -e .
