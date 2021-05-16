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

"""Sapientino environments using a "dict" state space."""

from scenario_objects import *
import gym
from gym.spaces import Dict, Discrete

#from gym_sapientino.core.states import SapientinoState
#from gym_sapientino.sapientino_env import Sapientino
from envs.custom_env_dir.custom_uav_env import UAVEnv


class UAVDictSpace(UAVEnv):
    """
    A UAV environment with a dictionary state space.

    The components of the space are:
    - UAV x coordinate (Discrete)
    - UAV y coordinate (Discrete)
    - The battery Level (Discrete)
    - A boolean to check whether the last action was a beep (Discrete)
    - The color of the current cell (Discrete)
    """

    def __init__(self, *args, **kwargs):
        """Initialize the dictionary space."""
        super().__init__(*args, **kwargs)

        self._x_space = Discrete(CELLS_COLS)
        self._y_space = Discrete(CELLS_ROWS)
        self._z_space = Discete(int( (MAX_UAV_HEIGHT-MIN_UAV_HEIGHT)/UAV_Z_STEP ) + 1 ) # --> '+1' is due to the fact that the UAV can be at the minimum height 
        self._battery_space = Discrete(ITERATIONS_PER_EPISODE) # --> There will be as battery levels as the number of iterations per episode
        self._beep_space = Discrete(2)
        self._color_space = Discrete(PRIORITY_NUM)

    @property
    def observation_space(self):
        """Get the observation space."""

        def get_agent_dict_space(i: int):
            #agent_config = self.configuration.agent_configs[i]
            d = {
                "x": self._x_space,
                "y": self._y_space,
                "beep": self._beep_space,
                "color": self._color_space,
            }
            if (DIMENSION_2D==False):
                d["z"] = self._z_space
            if (UNLIMITED_BATTERY==False):
                d["battery"] = self._battery_space
            '''
            if agent_config.differential:
                d["theta"] = self._theta_space
            '''
            return Dict(d)

        return gym.spaces.Tuple(
            tuple(map(get_agent_dict_space, range(N_UAVS)))
        )

    @property
    def current_cells(self):
        """Get the current cell."""

        return [self.cells_matrix[uav._occupied_cell_coords[0]][uav._occupied_cell_coords[1]] for uav in self.agents]

    @property
    def last_commands(self):
        """Get the list of last commands."""
        return self._last_commands

    def to_dict(self):
        """Encode into a dictionary."""
        return tuple(
            {
                "x": uav._x_coord,
                "y": uav._y_coord,
                "z": uav._z_coord,
                "battery": uav._battery_level,
                "beep": int(self.last_commands[i] == BEEP), # --> Aggiungi BEEP alla lista delle azioni in 'my_utils.py' --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                "color": self.current_cells[i] # --> Non è necessatio 'self.current_cells[i].encoded_color', perchè nel nostro 'my_utils' i colori sono in un dict in cui la chiave è il numero corrsipondente al colore e il valore è il colore stesso (e non viceversa come in Sapientino)
            }
            for i, uav in enumerate(self.agents)
        )

    def observe(self, state: SapientinoState):
        """Observe the state."""
        return state.to_dict()

    # Inserito in 'def step(..)' 'in custom_uav_env-py':
    '''
    def _force_border_constraints(self, uav: Robot) -> Tuple[float, Robot]:
        reward = 0.0
        x, y = uav._x_coord, uav._y_coord
        if not (0 <= uav._x_coord < CELLS_COLS):
            reward += self.config.reward_outside_grid # --> scegli un reward per quando l'uav esce fuori dalla griglia --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            x = int(clip(uav._x_coord, 0, CELLS_COLS - 1))
        if not (0 <= uav._y_coord < CELLS_ROWS):
            reward += self.config.reward_outside_grid # --> scegli un reward per quando l'uav esce fuori dalla griglia --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            y = int(clip(uav._y_coord, 0, CELLS_ROWS - 1))
        return reward, r.move(x, y)
    '''

    def _do_beep(self, uav, command, cells_matrix):
        reward = 0.0
        if command == command.BEEP:
            position = uav._x_coord, uav._y_coord
            cell = cells_matrix[position[1]][position[0]]
            cell.beep()
            if cell._priority!=None:
                if cell._priority not in self.priority_count:
                    self.priority_count[cell._priority] = 0
                self.priority_count[cell._priority] += 1
            if cell.bip_count >= 2:
                reward -= 1

        return reward