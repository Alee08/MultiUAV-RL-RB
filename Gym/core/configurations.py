# -*- coding: utf-8 -*-
#
# Copyright 2019-2020 Marco Favorito, Luca Iocchi
#
# ------------------------------
#
# This file is part of gym-sapientino.
#
# gym-sapientino is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gym-sapientino is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gym-sapientino.  If not, see <https://www.gnu.org/licenses/>.
#

"""Classes for the environment configurations."""
from dataclasses import dataclass
from typing import Tuple

import gym
from gym.spaces import Discrete, MultiDiscrete
from gym.spaces import Tuple as GymTuple

from gym_sapientino.core.types import (
    ACTION_TYPE,
    COMMAND_TYPES,
    DifferentialCommand,
    Direction,
    NormalCommand,
    color2int,
)


@dataclass(frozen=True)
class SapientinoAgentConfiguration:
    """Configuration for a single agent."""

    differential: bool = False

    @property
    def action_space(self) -> gym.spaces.Discrete:
        """Get the action space.."""
        if self.differential:
            return Discrete(len(DifferentialCommand))
        else:
            return Discrete(len(NormalCommand))

    def get_action(self, action: int) -> COMMAND_TYPES:
        """Get the action."""
        if self.differential:
            return DifferentialCommand(action)
        else:
            return NormalCommand(action)


@dataclass(frozen=True)
class SapientinoConfiguration:
    """A class to represent Sapientino configurations."""

    # game configurations
    agent_configs: Tuple[SapientinoAgentConfiguration, ...] = (
        SapientinoAgentConfiguration(),
    )
    rows: int = 5
    columns: int = 7
    reward_outside_grid: float = -1.0
    reward_duplicate_beep: float = -1.0
    reward_per_step: float = -0.01

    # rendering configurations
    offx: int = 40
    offy: int = 100
    radius: int = 5
    size_square: int = 40

    @property
    def nb_robots(self) -> int:
        """Get the number of robots."""
        return len(self.agent_configs)

    @property
    def agent_config(self) -> "SapientinoAgentConfiguration":
        """Get the agent configuration."""
        assert self.nb_robots == 1, "Can be called only in single-agent mode."
        return self.agent_configs[0]

    @property
    def observation_space(self) -> gym.spaces.Tuple:
        """Get the observation space."""

        def get_observation_space(agent_config):
            postfix = 2, self.nb_colors
            if agent_config.differential:
                return MultiDiscrete(
                    (self.columns, self.rows, Direction.nb_directions()) + postfix
                )
            return MultiDiscrete((self.columns, self.rows) + postfix)

        return GymTuple(tuple(map(get_observation_space, self.agent_configs)))

    @property
    def action_space(self) -> gym.spaces.Tuple:
        """Get the action space of the robots."""
        spaces = tuple(Discrete(ac.action_space.n) for ac in self.agent_configs)
        return gym.spaces.Tuple(spaces)

    @property
    def win_width(self) -> int:
        """Get the window width."""
        if self.columns > 10:
            return self.size_square * (self.columns - 10)
        else:
            return 480

    @property
    def win_height(self) -> int:
        """Get the window height."""
        if self.rows > 10:
            return self.size_square * (self.rows - 10)
        else:
            return 520

    @property
    def nb_theta(self):
        """Get the number of orientations."""
        return Direction.nb_directions()

    @property
    def nb_colors(self):
        """Get the number of colors."""
        return len(color2int)

    def get_action(self, action) -> ACTION_TYPE:
        """Get the action."""
        return [ac.get_action(a) for a, ac in zip(action, self.agent_configs)]
