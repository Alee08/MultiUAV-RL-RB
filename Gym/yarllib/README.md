<h1 align="center">
  <b>yarllib</b>
</h1>

<p align="center">
  <a href="https://pypi.org/project/yarllib">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/yarllib">
  </a>
  <a href="https://pypi.org/project/yarllib">
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/yarllib" />
  </a>
  <a href="">
    <img alt="PyPI - Status" src="https://img.shields.io/pypi/status/yarllib" />
  </a>
  <a href="">
    <img alt="PyPI - Implementation" src="https://img.shields.io/pypi/implementation/yarllib">
  </a>
  <a href="">
    <img alt="PyPI - Wheel" src="https://img.shields.io/pypi/wheel/yarllib">
  </a>
  <a href="https://github.com/marcofavorito/yarllib/blob/master/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/marcofavorito/yarllib">
  </a>
</p>
<p align="center">
  <a href="">
    <img alt="test" src="https://github.com/marcofavorito/yarllib/workflows/test/badge.svg">
  </a>
  <a href="">
    <img alt="lint" src="https://github.com/marcofavorito/yarllib/workflows/lint/badge.svg">
  </a>
  <a href="">
    <img alt="docs" src="https://github.com/marcofavorito/yarllib/workflows/docs/badge.svg">
  </a>
  <a href="https://codecov.io/gh/marcofavorito/yarllib">
    <img alt="codecov" src="https://codecov.io/gh/marcofavorito/yarllib/branch/master/graph/badge.svg?token=FG3ATGP5P5">
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


Yet Another Reinforcement Learning Library.

Status: **development**.

## Why?

I had the need for a RL library/framework that:
- was clearly and simply implemented, with good enough performances;
- highly focused on modularity, customizability and extendability;
- wasn't merely Deep Reinforcement Learning oriented.

I couldn't find an existing library that satisfied my needs; 
hence I decided to implement _yet another_ RL library.

For me it is also an opportunity to 
have a better understanding of the RL algorithms
and to appreciate the nuances that you can't find on a book.

If you find this repo useful for your research or your project,
I'd be very glad :-) don't hesitate to reach me out!

## What

The package is both:
- a _library_, because it provides off-the-shelf functionalities to
  set up an RL experiment;
- a _framework_, because you can compose your custom model by implementing
  the interfaces, override the default behaviours, or use the existing
  components as-is.   

You can find more details in the 
[documentation](https://marcofavorito.github.io/yarllib).

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

yarllib is released under the GNU Lesser General Public License v3.0 or later (LGPLv3+).

Copyright 2020 Marco Favorito

## Authors

- [Marco Favorito](https://marcofavorito.github.io/)