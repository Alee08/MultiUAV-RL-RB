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

"""Define basic types."""

from enum import Enum
from typing import Dict, Sequence, Union


class NormalCommand(Enum):
    """Enumeration for normal commands."""

    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    BEEP = 4
    NOP = 5

    def __str__(self) -> str:
        """Get the string representation."""
        cmd = NormalCommand(self.value)
        if cmd == NormalCommand.LEFT:
            return "<"
        elif cmd == NormalCommand.RIGHT:
            return ">"
        elif cmd == NormalCommand.UP:
            return "^"
        elif cmd == NormalCommand.DOWN:
            return "v"
        elif cmd == NormalCommand.BEEP:
            return "o"
        elif cmd == NormalCommand.NOP:
            return "_"
        else:
            raise ValueError("Shouldn't be here...")


class DifferentialCommand(Enum):
    """Enumeration for differential commands."""

    LEFT = 0
    FORWARD = 1
    RIGHT = 2
    BACKWARD = 3
    BEEP = 4
    NOP = 5

    def __str__(self) -> str:
        """Get the string representation."""
        cmd = DifferentialCommand(self.value)
        if cmd == DifferentialCommand.LEFT:
            return "<"
        elif cmd == DifferentialCommand.RIGHT:
            return ">"
        elif cmd == DifferentialCommand.FORWARD:
            return "^"
        elif cmd == DifferentialCommand.BACKWARD:
            return "v"
        elif cmd == DifferentialCommand.BEEP:
            return "o"
        elif cmd == DifferentialCommand.NOP:
            return "_"
        else:
            raise ValueError("Shouldn't be here...")


COMMAND_TYPES = Union[NormalCommand, DifferentialCommand]


class Direction(Enum):
    """A class to represent the four directions (up, down, left, right)."""

    RIGHT = 0
    UP = 90
    LEFT = 180
    DOWN = 270

    def rotate_left(self) -> "Direction":
        """Rotate the direction to the left."""
        th = (self.th + 90) % 360
        return Direction(th)

    def rotate_right(self) -> "Direction":
        """Rotate the direction to the right."""
        th = (self.th - 90) % 360
        if th == -90:
            th = 270
        return Direction(th)

    @staticmethod
    def nb_directions() -> int:
        """Get the number of allowed directions."""
        return len(Direction)

    @property
    def th(self):
        """Get the theta value of the direction."""
        return self.value


class Colors(Enum):
    """Enumeration for colors."""

    BLANK = "blank"
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    PINK = "pink"
    BROWN = "brown"
    GRAY = "gray"
    PURPLE = "purple"

    def __str__(self) -> str:
        """Get the string representation."""
        return self.value

    def __int__(self) -> int:
        """Get the integer representation."""
        return color2int[self]


color2int: Dict[Colors, int] = {c: i for i, c in enumerate(list(Colors))}

ACTION_TYPE = Sequence[COMMAND_TYPES]
