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

"""Objects of the game."""

import itertools
from typing import Dict, Tuple

from gym_sapientino.core.configurations import SapientinoConfiguration
from gym_sapientino.core.constants import TOKENS
from gym_sapientino.core.types import (
    COMMAND_TYPES,
    Colors,
    DifferentialCommand,
    Direction,
    NormalCommand,
    color2int,
)


class Robot:
    """A class to represent a robot."""

    def __init__(
        self, config: SapientinoConfiguration, x: int, y: int, th: Direction, id_: int
    ):
        """Initialize the robot."""
        self.config = config
        self._id = id_
        self.x = x
        self.y = y
        self.direction = th

    @property
    def id(self) -> int:
        """Get the robot's id."""
        return self._id

    @property
    def position(self) -> Tuple[int, int]:
        """Get the position."""
        return self.x, self.y

    def move(self, x: int, y: int) -> "Robot":
        """Move to a location."""
        return Robot(self.config, x, y, self.direction, self.id)

    def step(self, command: COMMAND_TYPES) -> "Robot":
        """Compute the next location."""
        if isinstance(command, NormalCommand):
            return self._step_normal(command)
        elif isinstance(command, DifferentialCommand):
            return self._step_differential(command)
        else:
            raise ValueError("Command not recognized.")

    def _step_normal(self, command: NormalCommand) -> "Robot":
        x, y = self.x, self.y
        if command == command.DOWN:
            y -= 1
        elif command == command.UP:
            y += 1
        elif command == command.RIGHT:
            x += 1
        elif command == command.LEFT:
            x -= 1
        return Robot(self.config, x, y, self.direction, self.id)

    def _step_differential(self, command: DifferentialCommand) -> "Robot":
        dx = 1 if self.direction.th == 0 else -1 if self.direction.th == 180 else 0
        dy = 1 if self.direction.th == 90 else -1 if self.direction.th == 270 else 0
        x, y = self.x, self.y
        direction = self.direction
        if command == command.LEFT:
            direction = direction.rotate_left()
        elif command == command.RIGHT:
            direction = direction.rotate_right()
        elif command == command.FORWARD:
            x += dx
            y += dy
        elif command == command.BACKWARD:
            x -= dx
            y -= dy
        return Robot(self.config, x, y, direction, self.id)

    @property
    def encoded_theta(self) -> int:
        """Encode the theta."""
        return self.direction.th // 90


class Cell:
    """A class to represent a cell on the grid."""

    def __init__(self, config: SapientinoConfiguration, x: int, y: int, color: Colors):
        """Initialize the cell."""
        self.config = config
        self.x = x
        self.y = y
        self.color = color
        self.bip_count = 0

    @property
    def encoded_color(self) -> int:
        """Encode the color."""
        return color2int[self.color]

    def beep(self) -> None:
        """Do a beep."""
        self.bip_count += 1


class BlankCell(Cell):
    """A blank cell."""

    def __init__(self, config: SapientinoConfiguration, x: int, y: int):
        """Initialize the blank cell."""
        super().__init__(config, x, y, Colors.BLANK)


class SapientinoGrid:
    """The grid of the Sapientino environment."""

    def __init__(self, config: SapientinoConfiguration):
        """Initialize the grid."""
        self.config = config

        self.rows = config.rows
        self.columns = config.columns

        self.cells: Dict[Tuple[int, int], Cell] = {}
        self.color_count: Dict[Colors, int] = {}

        self._populate_token_grid()

    def _populate_token_grid(self):
        # add color cells
        for t in TOKENS:
            x, y = t[2], t[3]
            color = t[1]
            color_cell = Cell(self.config, x, y, Colors(color))
            self.cells[(x, y)] = color_cell

        # add blank cells
        for x, y in itertools.product(
            range(self.config.columns), range(self.config.rows)
        ):
            if (x, y) not in self.cells:
                self.cells[(x, y)] = BlankCell(self.config, x, y)
