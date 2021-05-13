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

"""This module contains tests for the temprl/wrapper.py module."""
import numpy as np
import pytest
from flloat.parser.ldlf import LDLfParser
from flloat.semantics import PLInterpretation
from gym.spaces import Discrete, MultiDiscrete, Tuple

from temprl.automata import RewardDFA
from temprl.wrapper import TemporalGoal, TemporalGoalWrapper
from tests.utils import Action, GymTestEnv, GymTestObsWrapper


class TestWrapper:
    """Test that the wrapper works as expected."""

    @classmethod
    def setup_class(cls):
        """Set the tests up."""
        cls.env = GymTestEnv(n_states=5)

        cls.formula = LDLfParser()("<(!s4)*;s3;(!s4)*;s0;(!s4)*;s4>tt")
        cls.tg = TemporalGoal(
            formula=cls.formula,
            reward=10.0,
            labels={"s0", "s1", "s2", "s3", "s4"},
            reward_shaping=True,
            extract_fluents=None,
        )
        cls.wrapped = TemporalGoalWrapper(env=cls.env, temp_goals=[cls.tg])

    def test_observation_space(self):
        """Test that the combined observation space is computed as expected."""
        assert self.wrapped.observation_space == Tuple(
            (Discrete(5), MultiDiscrete([7]))
        )

    def test_temporal_goal_reward(self):
        """Test that the 'reward' property of the temporal goal works correctly."""
        assert self.tg.reward == 10.0

    def test_temporal_goal_formula(self):
        """Test that the 'formula' property of the temporal goal works correctly."""
        assert self.tg.formula == self.formula

    def test_temporal_goal_automaton(self):
        """Test that the 'automaton' property of the temporal goal works correctly."""
        assert isinstance(self.tg.automaton, RewardDFA)

    def test_extract_fluents_raises_exception(self):
        """Test that 'extract_fluents' raises 'NotImplementedError' if not set properly."""
        with pytest.raises(NotImplementedError):
            self.tg.extract_fluents(None, None)

    @classmethod
    def teardown_class(cls):
        """Tear the tests down."""


class TestWrapperFromDFA:
    """Test that the wrapper built from a DFA works as expected."""

    @classmethod
    def setup_class(cls):
        """Set the tests up."""
        cls.env = GymTestEnv(n_states=5)

        cls.formula = LDLfParser()("<(!s4)*;s3;(!s4)*;s0;(!s4)*;s4>tt")
        cls.tg = TemporalGoal(
            automaton=cls.formula.to_automaton(),
            reward=10.0,
            labels={"s0", "s1", "s2", "s3", "s4"},
            reward_shaping=True,
            extract_fluents=None,
        )
        cls.wrapped = TemporalGoalWrapper(env=cls.env, temp_goals=[cls.tg])

    def test_observation_space(self):
        """Test that the combined observation space is computed as expected."""
        assert self.wrapped.observation_space == Tuple(
            (Discrete(5), MultiDiscrete([7]))
        )

    def test_temporal_goal_reward(self):
        """Test that the 'reward' property of the temporal goal works correctly."""
        assert self.tg.reward == 10.0

    def test_temporal_goal_formula(self):
        """Test that the 'formula' property of the temporal goal works correctly."""
        assert self.tg.formula is None

    def test_temporal_goal_automaton(self):
        """Test that the 'automaton' property of the temporal goal works correctly."""
        assert isinstance(self.tg.automaton, RewardDFA)

    def test_extract_fluents_raises_exception(self):
        """Test that 'extract_fluents' raises 'NotImplementedError' if not set properly."""
        with pytest.raises(NotImplementedError):
            self.tg.extract_fluents(None, None)

    @classmethod
    def teardown_class(cls):
        """Tear the tests down."""


class TestWrapperRewardShaping:
    """This class tests the case when we apply reward shaping."""

    @classmethod
    def setup_class(cls):
        """Set the tests up."""
        cls.env = GymTestObsWrapper(n_states=5)
        cls.tg = TemporalGoal(
            formula=LDLfParser()("<(!s4)*;s3;(!s4)*;s0;(!s4)*;s4>tt"),
            reward=10.0,
            labels={"s0", "s1", "s2", "s3", "s4"},
            reward_shaping=True,
            zero_terminal_state=True,
            extract_fluents=lambda obs, action: PLInterpretation({"s" + str(obs[0])}),
        )
        cls.wrapped = TemporalGoalWrapper(env=cls.env, temp_goals=[cls.tg])

    def test_reward_shaping(self):
        """Test that the reward shaping works as expected."""
        obs = self.wrapped.reset()
        total_reward = 0
        assert np.array_equal(obs, [[0], [0]])
        assert not self.tg.is_true()
        assert not self.tg.is_failed()

        # s1
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s2
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s3 - positive reward
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert np.isclose(reward, 3.3333, rtol=1e-9, atol=0.0001)
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s2
        obs, reward, done, info = self.wrapped.step(Action.LEFT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s1
        obs, reward, done, info = self.wrapped.step(Action.LEFT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s0 - positive reward
        obs, reward, done, info = self.wrapped.step(Action.LEFT)
        total_reward += reward
        assert np.isclose(reward, 3.3333, rtol=1e-9, atol=0.0001)
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s1
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s2
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s3
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s4
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert np.isclose(reward, 4.3333, rtol=1e-9, atol=0.0001)
        assert total_reward == 11.0
        assert done
        assert not self.tg.is_failed()
        assert self.tg.is_true()

    def test_when_temporal_goal_is_failed(self):
        """Test the case when the temporal goal is failed."""
        obs = self.wrapped.reset()
        assert np.array_equal(obs, [[0], [0]])

        # s1
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        assert reward == 0
        # s2
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        assert reward == 0
        # s3 - positive reward
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        assert np.isclose(reward, 3.3333, rtol=1e-9, atol=0.0001)
        # s4 - temporal goal fails.
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        assert np.isclose(reward, 1 - 3.3333, rtol=1e-9, atol=0.0001)
        assert self.tg.is_failed()


class TestWrapperRewardShapingFromDFA:
    """This class tests reward shaping when we build the temporal goal from the DFA."""

    @classmethod
    def setup_class(cls):
        """Set the tests up."""
        cls.env = GymTestObsWrapper(n_states=5)
        cls.dfa = LDLfParser()("<(!s4)*;s3;(!s4)*;s0;(!s4)*;s4>tt").to_automaton(
            labels={"s0", "s1", "s2", "s3", "s4"}
        )
        cls.tg = TemporalGoal(
            automaton=cls.dfa,
            reward=10.0,
            reward_shaping=True,
            zero_terminal_state=True,
            extract_fluents=lambda obs, action: PLInterpretation({"s" + str(obs[0])}),
        )
        cls.wrapped = TemporalGoalWrapper(env=cls.env, temp_goals=[cls.tg])

    def test_reward_shaping(self):
        """Test that the reward shaping works as expected."""
        obs = self.wrapped.reset()
        total_reward = 0
        assert np.array_equal(obs, [[0], [0]])
        assert not self.tg.is_true()
        assert not self.tg.is_failed()

        # s1
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s2
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s3 - positive reward
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert np.isclose(reward, 3.3333, rtol=1e-9, atol=0.0001)
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s2
        obs, reward, done, info = self.wrapped.step(Action.LEFT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s1
        obs, reward, done, info = self.wrapped.step(Action.LEFT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s0 - positive reward
        obs, reward, done, info = self.wrapped.step(Action.LEFT)
        total_reward += reward
        assert np.isclose(reward, 3.3333, rtol=1e-9, atol=0.0001)
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s1
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s2
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s3
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s4
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert np.isclose(reward, 4.3333, rtol=1e-9, atol=0.0001)
        assert total_reward == 11.0
        assert done
        assert not self.tg.is_failed()
        assert self.tg.is_true()

    def test_when_temporal_goal_is_failed(self):
        """Test the case when the temporal goal is failed."""
        obs = self.wrapped.reset()
        assert np.array_equal(obs, [[0], [0]])

        # s1
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        assert reward == 0
        # s2
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        assert reward == 0
        # s3 - positive reward
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        assert np.isclose(reward, 3.3333, rtol=1e-9, atol=0.0001)
        # s4 - temporal goal fails.
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        assert np.isclose(reward, 1 - 3.3333, rtol=1e-9, atol=0.0001)
        assert self.tg.is_failed()


class TestWrapperNoRewardShaping:
    """This class tests the case when we don't apply reward shaping."""

    @classmethod
    def setup_class(cls):
        """Set the tests up."""
        cls.env = GymTestObsWrapper(n_states=5)
        cls.tg = TemporalGoal(
            formula=LDLfParser()("<(!s4)*;s3;(!s4)*;s0;(!s4)*;s4>tt"),
            reward=10.0,
            labels={"s0", "s1", "s2", "s3", "s4"},
            reward_shaping=False,
            extract_fluents=lambda obs, action: PLInterpretation({"s" + str(obs[0])}),
        )
        cls.wrapped = TemporalGoalWrapper(env=cls.env, temp_goals=[cls.tg])

    def test_no_reward_shaping(self):
        """Test that the reward shaping works as expected."""
        self.wrapped.temp_goals[0]._simulator.reward_shaping = False
        obs = self.wrapped.reset()
        total_reward = 0
        assert np.array_equal(obs, [[0], [0]])
        assert not self.tg.is_true()
        assert not self.tg.is_failed()

        # s1
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s2
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s3 - positive reward
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s2
        obs, reward, done, info = self.wrapped.step(Action.LEFT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s1
        obs, reward, done, info = self.wrapped.step(Action.LEFT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s0 - positive reward
        obs, reward, done, info = self.wrapped.step(Action.LEFT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s1
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s2
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s3
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s4
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert np.isclose(reward, 11.0, rtol=1e-9, atol=0.0001)
        assert total_reward == 11.0
        assert done
        assert not self.tg.is_failed()
        assert self.tg.is_true()

    def test_when_temporal_goal_is_failed(self):
        """Test the case when the temporal goal is failed."""
        obs = self.wrapped.reset()
        assert np.array_equal(obs, [[0], [0]])

        # s1
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        assert reward == 0
        # s2
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        assert reward == 0
        # s3 - positive reward
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        assert reward == 0
        # s4 - temporal goal fails.
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        assert np.isclose(reward, 1.0)
        assert self.tg.is_failed()


class TestWrapperRewardShapingWithZeroTerminalState:
    """This class tests the case when we apply reward shaping + zero terminal state."""

    @classmethod
    def setup_class(cls):
        """Set the tests up."""
        cls.env = GymTestObsWrapper(n_states=5)
        cls.tg = TemporalGoal(
            formula=LDLfParser()("<(!s4)*;s3;(!s4)*;s0;(!s4)*;s4>tt"),
            reward=10.0,
            labels={"s0", "s1", "s2", "s3", "s4"},
            reward_shaping=True,
            zero_terminal_state=True,
            extract_fluents=lambda obs, action: PLInterpretation({"s" + str(obs[0])}),
        )
        cls.wrapped = TemporalGoalWrapper(env=cls.env, temp_goals=[cls.tg])

    def test_reward_shaping(self):
        """Test that the reward shaping works as expected."""
        obs = self.wrapped.reset()
        total_reward = 0
        assert np.array_equal(obs, [[0], [0]])
        assert not self.tg.is_true()
        assert not self.tg.is_failed()

        # s1
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s2
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s3 - positive reward
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert np.isclose(reward, 3.3333, rtol=1e-9, atol=0.0001)
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s2
        obs, reward, done, info = self.wrapped.step(Action.LEFT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s1
        obs, reward, done, info = self.wrapped.step(Action.LEFT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s0 - positive reward
        obs, reward, done, info = self.wrapped.step(Action.LEFT)
        total_reward += reward
        assert np.isclose(reward, 3.3333, rtol=1e-9, atol=0.0001)
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s1
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s2
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s3
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s4
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        total_reward += reward
        assert np.isclose(reward, 4.3333, rtol=1e-9, atol=0.0001)
        assert total_reward == 11.0
        assert done
        assert not self.tg.is_failed()
        assert self.tg.is_true()

    def test_when_temporal_goal_is_failed(self):
        """Test the case when the temporal goal is failed."""
        obs = self.wrapped.reset()
        assert np.array_equal(obs, [[0], [0]])

        # s1
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        assert reward == 0
        # s2
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        assert reward == 0
        # s3 - positive reward
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        assert np.isclose(reward, 3.3333, rtol=1e-9, atol=0.0001)
        # s4 - temporal goal fails.
        obs, reward, done, info = self.wrapped.step(Action.RIGHT)
        assert np.isclose(reward, 1 - 3.3333, rtol=1e-9, atol=0.0001)
        assert self.tg.is_failed()
