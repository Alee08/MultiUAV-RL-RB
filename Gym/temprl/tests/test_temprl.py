# -*- coding: utf-8 -*-
#
# Copyright 2020 Marco Favorito
#
# ------------------------------
#
# This file is part of temprl.
#
# temprl is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# temprl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with temprl.  If not, see <https://www.gnu.org/licenses/>.
#

"""Tests for `temprl` package."""

import numpy as np
from flloat.parser.ldlf import LDLfParser
from flloat.semantics import PLInterpretation
from gym.spaces import MultiDiscrete

from temprl.wrapper import TemporalGoal, TemporalGoalWrapper
from tests.utils import GymTestEnv, q_function_learn, q_function_test, wrap_observation


class TestSimpleEnv:
    """This class contains tests for a simple Gym environment."""

    @classmethod
    def setup_class(cls):
        """Set the tests up."""
        cls.env = GymTestEnv(n_states=5)
        cls.Q = q_function_learn(cls.env, nb_episodes=200)

    def test_optimal_policy(self):
        """Test that the optimal policy maximizes the reward."""
        history = q_function_test(self.env, self.Q, nb_episodes=10)
        assert np.isclose(np.average(history), 1.0)

    @classmethod
    def teardown_class(cls):
        """Tear the tests down."""


class TestTempRLWithSimpleEnv:
    """This class contains tests for a Gym wrapper with a simple temporal goal."""

    @classmethod
    def setup_class(cls):
        """Set the tests up."""
        cls.env = GymTestEnv(n_states=5)
        cls.tg = TemporalGoal(
            formula=LDLfParser()("<(!s4)*;s3;(!s4)*;s0;(!s4)*;s4>tt"),
            reward=10.0,
            labels={"s0", "s1", "s2", "s3", "s4"},
            reward_shaping=True,
            extract_fluents=lambda obs, action: PLInterpretation({"s" + str(obs)}),
        )
        cls.wrapped = TemporalGoalWrapper(env=cls.env, temp_goals=[cls.tg])
        # from (s, [q]) to (s, q)
        n, q = cls.env.observation_space.n, cls.tg.observation_space.n
        cls.wrapped = wrap_observation(
            cls.wrapped, MultiDiscrete([n, q]), lambda o: (o[0], o[1][0])
        )
        cls.Q = q_function_learn(cls.wrapped, nb_episodes=5000)

    def test_learning_wrapped_env(self):
        """Test that learning with the unwrapped env is feasible."""
        history = q_function_test(self.wrapped, self.Q, nb_episodes=10)
        print("Reward history: {}".format(history))
        assert np.isclose(np.average(history), 11.0)

    @classmethod
    def teardown_class(cls):
        """Tear the tests down."""
