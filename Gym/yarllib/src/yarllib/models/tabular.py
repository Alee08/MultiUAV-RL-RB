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

"""This module contains simple implementations of tabular RL algorithms."""

from abc import ABC
from typing import Any

import numpy as np
from gym.spaces import Discrete

from yarllib.core import Model
from yarllib.helpers.base import SparseTable
from yarllib.types import Action, AgentObservation, State


def _make_table(nb_rows: int, nb_cols: int, sparse: bool):
    """Make a table to store the Q-function."""
    if sparse:
        return SparseTable(nb_rows, nb_cols)
    else:
        return np.random.rand(nb_rows, nb_cols) * np.finfo(float).eps


class TabularModel(Model, ABC):
    """A tabular model."""

    def __init__(
        self,
        state_space: Discrete,
        action_space: Discrete,
        alpha: float = 0.1,
        gamma: float = 0.99,
        sparse: bool = False,
    ):
        """
        Initialize a tabular model.

        :param state_space: the state space.
        :param action_space: the action space.
        :param alpha: the learning rate.
        :param gamma: the gamma parameter.
        """
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.q = _make_table(state_space.n, action_space.n, sparse)

    def get_best_action(self, state: State) -> Any:
        """Get the best action."""
        return self.q[state].argmax()


class TabularQLearning(TabularModel):
    """Q-Learning model."""

    def on_step_end(self, step, agent_observation: AgentObservation, **kwargs):
        """On step end event."""
        self.update(agent_observation)

    def update(self, agent_observation: AgentObservation):
        """Update the model."""
        q = self.q
        gamma = self.gamma
        alpha = self.alpha
        s, a, r, sp, done = agent_observation
        q[s, a] = q[s, a] + alpha * (r + gamma * q[sp].max() - q[s, a])


class TabularSarsa(TabularModel):
    """Sarsa model."""

    def __init__(self, *args, **kwargs):
        """Initialize the model."""
        super().__init__(*args, **kwargs)
        self._next_action: Action = None

    def get_action(self, state: State) -> Any:
        """Get the next action."""
        if self.context.is_training and self._next_action is not None:
            return self._next_action
        else:
            return self.context.policy.get_action(state)

    def on_episode_begin(self, *args, **kwargs):
        """On episode begin event."""
        self._next_action = None

    def on_step_end(self, step, agent_observation: AgentObservation, **kwargs):
        """On step end event."""
        _, _, _, sp, _ = agent_observation
        self._next_action = self.context.policy.get_action(sp)
        self.update(agent_observation)

    def update(self, agent_observation: AgentObservation):
        """Update the model."""
        q = self.q
        gamma = self.gamma
        alpha = self.alpha
        ap = self._next_action
        s, a, r, sp, done = agent_observation
        q[s, a] = q[s, a] + alpha * (r + gamma * q[sp, ap] - q[s, a])
