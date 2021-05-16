# -*- coding: utf-8 -*-
#
# Copyright 2020 Marco Favorito
#
# ------------------------------
#
# This file is part of yarllib.
#
# yarllib is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# yarllib is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with yarllib.  If not, see <https://www.gnu.org/licenses/>.
#

"""Test helpers."""

import itertools

import numpy as np
import pytest
from gym.spaces import MultiDiscrete

from yarllib.helpers.encoding import Encoder
from yarllib.helpers.history import History, history_from_json, history_to_json


@pytest.mark.parametrize("space", [(5, 2), (5, 2, 3), (10, 20), (20, 10), (42, 1, 42)])
def test_encoder_with_sampling(space):
    """Test space_encoder with sampling."""
    NUM_SAMPLES = int(np.prod(space))
    x = MultiDiscrete(space)
    e = Encoder(x)

    for _ in range(NUM_SAMPLES):
        i = x.sample()
        enc = e.encode(i)
        dec = e.decode(enc)
        assert np.equal(i, dec).all()


def test_encoder_codomain():
    """Test that the function computed is surjective."""
    dims = (42, 2, 10)
    space = MultiDiscrete(dims)
    e = Encoder(space)

    codomain = set()
    for el in itertools.product(*map(lambda x: set(range(x)), dims)):
        enc = e.encode(np.array(el))
        codomain.add(enc)

    assert len(codomain) == space.nvec.prod()
    assert codomain == set(range(len(codomain)))


def test_history_json_serialization():
    """Test History serialization."""
    observations = [(0, 1, 0.0, 1), (1, 1, 0.0, 0)]
    expected_history = History([observations])
    actual_history = history_from_json(history_to_json(expected_history))

    for e, a in zip(expected_history.episodes, actual_history.episodes):
        assert (e.observations == a.observations).all()
