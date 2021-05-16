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

"""Helpers module."""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from yarllib.helpers.base import array_to_list
from yarllib.types import Action, Reward, State

AgentObs = Tuple[State, Action, Reward, State]
EpisodeAgentObs = List[AgentObs]


class EpisodeHistory:
    """Class that represents a history of observation in one episode."""

    def __init__(self, observations: List[AgentObs]) -> None:
        """
        Initialize the episode history.

        :param observations: the list of (s, a, r, s') tuples.
        """
        # copy the argument
        self._observations = np.array(observations, copy=True, dtype=object)

    @property
    def length(self) -> int:
        """Get the length of the episode."""
        return self._observations.shape[0]

    @property
    def rewards(self):
        """Get the rewards."""
        return self._observations[:, 2]

    @property
    def observations(self) -> np.ndarray:
        """Get the observations as numpy arrays (i.e. length x 4 array)."""
        return self._observations

    @property
    def average_reward(self) -> float:
        """Get the average reward."""
        return self._observations[:, 2].mean()

    @property
    def total_reward(self) -> float:
        """Get the total reward of an episode."""
        return self._observations[:, 2].sum()


class History:
    """A history of a session."""

    def __init__(
        self,
        episodes: List[EpisodeAgentObs],
        is_training: Optional[bool] = None,
        seed: Optional[int] = None,
        name: str = "",
    ):
        """
        Initialize a history of episodes.

        :param episodes: a list of episode history.
                       | Each episode history is a list of tuples (s, a, r, s').
        :param is_training: whether the agent was in training mode or test mode.
        :param seed: the seed to reproduce the experiments.
        :param name: the name of the experiment.
        """
        self._episodes = np.asarray(list(map(EpisodeHistory, episodes)))
        self._is_training = is_training
        self._seed = seed
        self._name = name

    @property
    def seed(self) -> Optional[int]:
        """Get the seed."""
        return self._seed

    @property
    def is_training(self) -> Optional[int]:
        """Get the seed."""
        return self._is_training

    @property
    def episodes(self) -> Sequence[EpisodeHistory]:
        """Get the list of episode histories."""
        return self._episodes

    @property
    def name(self) -> str:
        """Get the name."""
        return self._name

    @property
    def nb_episodes(self) -> int:
        """Get the number of episodes."""
        return self._episodes.shape[0]

    @property
    def average_rewards(self):
        """Get the average rewards of every episode."""
        return np.asarray(
            [episode_history.average_reward for episode_history in self._episodes]
        )

    @property
    def total_rewards(self):
        """Get the average rewards of every episode."""
        return np.asarray(
            [episode_history.total_reward for episode_history in self._episodes]
        )

    @property
    def lengths(self):
        """Get the lengths of the episodes."""
        return np.asarray(
            [episode_history.length for episode_history in self._episodes]
        )


def history_to_json(h: History) -> Dict:
    """Transform a history object to JSON."""
    return dict(
        seed=h.seed,
        is_training=h.is_training,
        episodes=[episode_history_to_json(e) for e in h.episodes],
    )


def episode_history_to_json(e: EpisodeHistory) -> Any:
    """Transform a episode history to JSON."""
    return dict(
        observations=array_to_list(e.observations[:, 0]),
        actions=array_to_list(e.observations[:, 1]),
        rewards=e.rewards.tolist(),
        last_observation=array_to_list(e.observations[-1][-1]),
    )


def history_from_json(o: Dict) -> History:
    """Return a history object from JSON."""
    assert set(o.keys()) == {"seed", "is_training", "episodes"}
    is_training = o["is_training"]
    seed = o["seed"]
    episodes: List[EpisodeAgentObs] = [
        agent_observations_from_json(e) for e in o["episodes"]
    ]
    return History(episodes=episodes, is_training=is_training, seed=seed)


def agent_observations_from_json(o: Dict) -> EpisodeAgentObs:
    """Return a list of agent observations from JSON."""
    observations = o["observations"]
    actions = o["actions"]
    rewards = o["rewards"]
    last_observation = o["last_observation"]
    n = len(observations)
    episode_observations: EpisodeAgentObs = []
    for i in range(n - 1):
        episode_observations.append(
            (observations[i], actions[i], rewards[i], observations[i + 1])
        )
    episode_observations.append(
        (observations[n - 1], actions[n - 1], rewards[n - 1], last_observation)
    )
    return episode_observations


def episode_history_from_json(o: Dict) -> EpisodeHistory:
    """Return an episode history object from JSON."""
    agent_obs = agent_observations_from_json(o)
    return EpisodeHistory(agent_obs)
