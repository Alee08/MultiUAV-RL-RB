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

"""This module contains the implementation of common RL policies."""

import random
from typing import cast

import gym

from yarllib.core import Policy
from yarllib.types import State


class RandomPolicy(Policy):
    """A random policy."""

    def __init__(self, action_space: gym.Space):
        """
        Initialize a random policy.

        :param action_space: the action space to take decisions on.
        """
        super().__init__()
        self._action_space = action_space

    def get_action(self, state: State):
        """
        Get the action at random.

        :param state: the state from where to take the decision.
        :return: the action.
        """
        return cast(gym.spaces.Space, self._action_space).sample()


class GreedyPolicy(Policy):
    """A greedy policy."""

    def get_action(self, state):
        """
        Get the action greedily (according to the model).

        :param state: the state from where to take the decision.
        :return: the greedy action.
        """
        return self.model.get_best_action(state)


class EpsGreedyPolicy(Policy):
    """A epsilon-greedy policy."""

    def __init__(self, epsilon: float = 0.1):
        """
        Initialize an epsilon-greedy policy.

        :param epsilon: the epsilon parameter.
        """
        super().__init__()
        self.epsilon = epsilon

    def get_action(self, state: State):
        """
        Get the action random with probability epsilon, greedily otherwise.

        :param state: the state from where to take the decision.
        :return: the greedy or random action.
        """
        if random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            return self.model.get_best_action(state)
