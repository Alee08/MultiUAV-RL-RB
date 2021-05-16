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

"""Constants of the game."""

from gym_sapientino.core.types import Colors

TOKENS = [
    ["r1", Colors.RED.value, 0, 0],
    ["r2", Colors.RED.value, 1, 1],
    ["r3", Colors.RED.value, 6, 3],
    ["g1", Colors.GREEN.value, 4, 0],
    ["g2", Colors.GREEN.value, 5, 2],
    ["g3", Colors.GREEN.value, 5, 4],
    ["b1", Colors.BLUE.value, 1, 3],
    ["b2", Colors.BLUE.value, 2, 4],
    ["b3", Colors.BLUE.value, 6, 0],
    ["p1", Colors.PINK.value, 2, 1],
    ["p2", Colors.PINK.value, 2, 3],
    ["p3", Colors.PINK.value, 4, 2],
    ["n1", Colors.BROWN.value, 3, 0],
    ["n2", Colors.BROWN.value, 3, 4],
    ["n3", Colors.BROWN.value, 6, 1],
    ["y1", Colors.GRAY.value, 0, 2],
    ["y2", Colors.GRAY.value, 3, 1],
    ["y3", Colors.GRAY.value, 4, 3],
    ["u1", Colors.PURPLE.value, 0, 4],
    ["u2", Colors.PURPLE.value, 1, 0],
    ["u3", Colors.PURPLE.value, 5, 1],
]

black = [0, 0, 0]
white = [255, 255, 255]
grey = [180, 180, 180]
orange = [180, 100, 20]
red = [180, 0, 0]
