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

"""Test utils."""
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, Optional

import gym
import numpy as np
from gym.spaces import Discrete, MultiDiscrete


class Action(Enum):
    """An enum to describe the available actions."""

    NOP = 0
    LEFT = 1
    RIGHT = 2

    def get_delta(self) -> int:
        """Get the state change."""
        return 0 if self == self.NOP else -1 if self == self.LEFT else 1


class GymTestEnv(gym.Env):
    """
    A class that implements a simple Gym environment.

    It is a chain-like environment:
    - the state space is the set of cells in a row;
    - the action space is {LEFT, RIGHT, NOP}.
    - a reward signal +1 is given if the right-most state is reached.
    - a reward signal -1 is given if an illegal move is done
      (i.e. LEFT in the left-most state of the chain).
    - the game ends when a time limit is reached or the agent reaches the right-most state.
    """

    def __init__(self, n_states: int = 2):
        """Initialize the Gym test environment."""
        assert n_states >= 2
        self._current_state = 0
        self._last_action = None  # type: Optional[int]
        self.n_states = n_states
        self.max_steps = self.n_states * 10

        self.observation_space = Discrete(self.n_states)
        self.actions = {
            Action.LEFT.value,
            Action.NOP.value,
            Action.RIGHT.value,
        }
        self.action_space = Discrete(len(self.actions))
        self.counter = 0

    def step(self, action: int):
        """Do a step in the environment."""
        self.counter += 1
        self._last_action = action
        reward = 0
        delta = Action(action).get_delta()
        if self._current_state + delta < 0:
            self._current_state = 0
            reward = -1
        elif self._current_state + delta >= self.n_states:
            self._current_state = self.n_states - 1
            reward = -1
        else:
            self._current_state += delta

        if self._current_state == self.n_states - 1:
            done = True
            reward = 1
        elif self.counter >= self.max_steps:
            done = True
        else:
            done = False

        return self._current_state, reward, done, {}

    def reset(self):
        """Reset the Gym env."""
        self._current_state = 0
        self.counter = 0
        self._last_action = None
        return self._current_state

    def render(self, mode="human"):
        """Render the current state of the environment."""
        print(
            "Current state={}, action={}".format(self._current_state, self._last_action)
        )


class GymTestObsWrapper(gym.ObservationWrapper):
    """
    This class is an observation wrapper for the GymTestEnv.

    It just makes the observation space an instance of MultiDiscrete
    rather than Discrete.
    """

    def __init__(self, n_states: int = 2):
        """Initialize the wrapper."""
        super().__init__(GymTestEnv(n_states))

        self.observation_space = MultiDiscrete((self.env.observation_space.n,))

    def observation(self, observation):
        """Wrap the observation."""
        return np.asarray([observation])


def q_function_learn(
    env: gym.Env, nb_episodes=100, alpha=0.1, eps=0.1, gamma=0.9
) -> Dict[Any, np.ndarray]:
    """Learn a Q-function from a Gym env using vanilla Q-Learning.

    :return the Q function: a dictionary from states to array of Q values for every action.
    """
    nb_actions = env.action_space.n
    Q: Dict[Any, np.ndarray] = defaultdict(
        lambda: np.random.randn(
            nb_actions,
        )
        * 0.01
    )

    def choose_action(state):
        if np.random.random() < eps:
            return np.random.randint(0, nb_actions)
        else:
            return np.argmax(Q[state])

    for _ in range(nb_episodes):
        state = env.reset()
        done = False
        while not done:
            action = choose_action(state)
            state2, reward, done, info = env.step(action)
            Q[state][action] += alpha * (
                reward + gamma * np.max(Q[state2]) - Q[state][action]
            )
            state = state2

    return Q


def q_function_test(
    env: gym.Env, Q: Dict[Any, np.ndarray], nb_episodes=10
) -> np.ndarray:
    """Test a Q-function against a Gym env.

    :return a list of rewards collected for every episode.
    """
    rewards = np.array([])
    for _ in range(nb_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = np.argmax(Q[state])
            state2, reward, done, info = env.step(action)
            total_reward += reward
            state = state2

        rewards = np.append(rewards, total_reward)
    return rewards


def wrap_observation(
    env: gym.Env, observation_space: gym.spaces.Space, observe: Callable
):
    """Wrap a Gym environment with an observation wrapper."""

    class _wrapper(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = observation_space

        def observation(self, observation):
            return observe(observation)

    return _wrapper(env)
