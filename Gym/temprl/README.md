<h1 align="center">
  <b>temprl</b>
</h1>

<p align="center">
  <a href="https://pypi.org/project/temprl">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/temprl">
  </a>
  <a href="https://pypi.org/project/temprl">
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/temprl" />
  </a>
  <a href="">
    <img alt="PyPI - Status" src="https://img.shields.io/pypi/status/temprl" />
  </a>
  <a href="">
    <img alt="PyPI - Implementation" src="https://img.shields.io/pypi/implementation/temprl">
  </a>
  <a href="">
    <img alt="PyPI - Wheel" src="https://img.shields.io/pypi/wheel/temprl">
  </a>
  <a href="https://github.com/whitemech/temprl/blob/master/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/whitemech/temprl">
  </a>
</p>
<p align="center">
  <a href="">
    <img alt="test" src="https://github.com/whitemech/temprl/workflows/test/badge.svg">
  </a>
  <a href="">
    <img alt="lint" src="https://github.com/whitemech/temprl/workflows/lint/badge.svg">
  </a>
  <a href="">
    <img alt="docs" src="https://github.com/whitemech/temprl/workflows/docs/badge.svg">
  </a>
  <a href="https://codecov.io/gh/whitemech/temprl">
    <img alt="codecov" src="https://codecov.io/gh/whitemech/temprl/branch/master/graph/badge.svg?token=FG3ATGP5P5">
  </a>
</p>
<p align="center">
  <a href="https://img.shields.io/badge/flake8-checked-blueviolet">
    <img alt="" src="https://img.shields.io/badge/flake8-checked-blueviolet">
  </a>
  <a href="https://img.shields.io/badge/mypy-checked-blue">
    <img alt="" src="https://img.shields.io/badge/mypy-checked-blue">
  </a>
  <a href="https://img.shields.io/badge/code%20style-black-black">
    <img alt="black" src="https://img.shields.io/badge/code%20style-black-black" />
  </a>
  <a href="https://www.mkdocs.org/">
    <img alt="" src="https://img.shields.io/badge/docs-mkdocs-9cf">
  </a>
</p>

Framework for Reinforcement Learning with Temporal Goals defined by LTLf/LDLf formulas.

Status: **development**.

## Install

Install the package:

- from PyPI:


        pip3 install temprl

- with `pip` from GitHub:


        pip3 install git+https://github.com/sapienza-rl/temprl.git


- or, clone the repository and install:


        git clone htts://github.com/sapienza-rl/temprl.git
        cd temprl
        pip install .


## Tests

To run tests: `tox`

To run only the code tests: `tox -e py3.7`

To run only the linters: 
- `tox -e flake8`
- `tox -e mypy`
- `tox -e black-check`
- `tox -e isort-check`

Please look at the `tox.ini` file for the full list of supported commands. 

## Docs

To build the docs: `mkdocs build`

To view documentation in a browser: `mkdocs serve`
and then go to [http://localhost:8000](http://localhost:8000)

## License

temprl is released under the GNU Lesser General Public License v3.0 or later (LGPLv3+).

Copyright 2018-2020 Marco Favorito

## Authors

- [Marco Favorito](https://whitemech.github.io/)
