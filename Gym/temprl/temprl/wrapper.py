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

"""Main module."""
import logging
from abc import ABC
from datetime import time
from time import sleep
from typing import Callable, List, Optional, Set

import gym
from flloat.semantics import PLInterpretation
from gym.spaces import Discrete, MultiDiscrete
from gym.spaces import Tuple as GymTuple
from pythomata.base import State, Symbol
from pythomata.dfa import DFA
from my_utils import *
#import graphviz
from temprl.temprl.automata import RewardAutomatonSimulator, RewardDFA, TemporalLogicFormula
#from envs.custom_env_dir.custom_uav_env import UAVEnv
#from env_wrapper import *

logger = logging.getLogger(__name__)


class TemporalGoal(ABC):
    """Abstract class to represent a temporal goal."""

    def __init__(
        self,
        formula: Optional[TemporalLogicFormula] = None,
        reward: float = 1.0,
        automaton: Optional[DFA] = None,
        labels: Optional[Set[Symbol]] = None,
        reward_shaping: bool = True,
        extract_fluents: Optional[Callable] = None,
        zero_terminal_state: bool = False,
        one_hot_encoding: bool = False,
    ):
        """
        Initialize a temporal goal.

        :param formula: the formula to be satisfied. it will be ignored if automaton is set.
        :param automaton: the pythomata.DFA instance. it will be
                        | the preferred input against 'formula'.
        :param reward: the reward associated to the temporal goal.
        :param labels: the set of all possible fluents
                     | (used to generate the full automaton).
        :param reward_shaping: the set of all possible fluents
                             | (used to generate the full automaton).
        :param extract_fluents: a callable that takes an observation
                             | and an actions, and returns a
                             | propositional interpretation with the active fluents.
                             | if None, the 'extract_fluents' method is taken.
        :param zero_terminal_state: when reward_shaping is True, make the
                                  | potential function at a terminal state equal to zero.
        :param one_hot_encoding: use one-hot encoding for representing the
                               | automata dimensions.
        """
        #DFA.to_dot(self, "/home/alee8", Optional[DFA])
        #DFA.to_dot(self, "/home/alee8/", automaton)
        if formula is None and automaton is None:
            raise ValueError("Provide either a formula or an automaton.")
        #DFA.to_dot(self, "/home/alee8", automaton)
        self._formula = formula
        if automaton:
            self._automaton = RewardDFA(automaton, reward)
        else:
            self._automaton = RewardDFA.from_formula(
                self._formula, reward, alphabet=labels
            )
        self._simulator = RewardAutomatonSimulator(
            self._automaton,
            reward_shaping=reward_shaping,
            zero_terminal_state=zero_terminal_state,
        )

        self._reward = reward
        self._one_hot_encoding = one_hot_encoding
        self._extract_fluents: Optional[Callable] = extract_fluents
        #print(automaton, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaooooooooooooooooooooooooooooooo")



    @property
    def observation_space(self) -> Discrete:
        """Return the observation space of the temporal goal."""
        # we add one virtual state for the 'super' sink state
        # - that is, when the symbol is not in the alphabet.
        # This is going to be a temporary workaround due to
        # the Pythomata's lack of support for this corner case.
        return Discrete(len(self._automaton.states) + 1)

    @property
    def formula(self):
        """Get the formula."""
        return self._formula

    @property
    def automaton(self):
        """Get the automaton."""

        return self._automaton

    @property
    def reward(self):
        """Get the reward."""
        return self._reward

    def extract_fluents(self, obs, action) -> PLInterpretation:
        #print(obs, action, "extract_fluents")

        """
        Extract high-level features from the observation.

        :return: the list of active fluents.
        """
        if self._extract_fluents is None:
            raise NotImplementedError
        return self._extract_fluents(obs, action)

    def step(self, observation, action) -> Optional[State]:
        #print("step - obs, action:", observation, action)
        """Do a step in the simulation."""
        fluents = self.extract_fluents(observation, action)

        #print("fluennnnnttsss1", fluents)
        #print(fluents, "fluentssssssssssssssssssssssssssss")
        self._simulator.step(fluents)
        #print("aaaaaaaaaaaaoooooooooooooooooooo",self._simulator.step(fluents))
        result = (
            self._simulator.cur_state
            if self._simulator.cur_state is not None
            else len(self._simulator.dfa.states)
        )
        #self._automaton.to_graphviz().render("filename")
        #DFA.to_dot(self, "/home/alee8", self._simulator.dfa.states)
        #print(result, "automatonautomatonautomatonautomatonautomatonautomaton")
        return result

    def reset(self):
        """Reset the simulation."""
        #print("Resetttttttttttttttttttttttttttttttttttt", self._simulator.cur_state)
        #breakpoint()
        self._simulator.reset()
        #print("dopppppppppppppppppppppppppppppppppppppp", self._simulator.cur_state)
        return self._simulator.cur_state

    def observe_reward(self, is_terminal_state: bool = False) -> float:
        """Observe the reward of the last transition."""
        return self._simulator.observe_reward(is_terminal_state)

    def is_true(self):
        """Check if the simulation is in a final state."""
        return self._simulator.is_true()

    def is_failed(self):
        """Check whether the simulation has failed."""
        return self._simulator.is_failed()


class TemporalGoalWrapper(gym.Wrapper):
    """Gym wrapper to include a temporal goal in the environment."""

    def __init__(
        self,
        env: gym.Env, # I pass my env., i.e. "UAVEnv"
        temp_goals: List[TemporalGoal],
    ):

        """
        Wrap a Gym environment with a temporal goal.

        :param env: the Gym environment to wrap.
        :param temp_goals: the temporal goal to be learnt
        """
        super().__init__(env)
        self.temp_goals = temp_goals
        self.observation_space = self._get_observation_space()

    def render(self, where_to_save=None, episode=None, how_often_render=None):
        self.env.render(where_to_save, episode, how_often_render)




    def _get_observation_space(self) -> gym.spaces.Space:
        """Return the observation space."""
        #print("self.env.observation_space", self.env.observation_space)
        #breakpoint()
        temp_goals_shape = tuple(tg.observation_space.n for tg in self.temp_goals)
        return GymTuple((self.env.observation_space, MultiDiscrete(temp_goals_shape)))

    def step_agent(self, agent, action):  # step_agent(self, agent, action)
        """Do a step in the Gym environment."""
        obs, reward, done, info = self.env.step_agent(agent, action)  # super().step(agent, action)
        if (UNLIMITED_BATTERY == True):
            #print("obs[2]", obs[2]._priority)
            #color_idx = get_color_id(obs[2]._priority)
            color_idx = obs[2]

            if color_idx == None:
                color_idx = 0

            #print("obs_dict", obs)

            #AGGIUSTARE. AGGIUNGERE LE PRIORITY NALLA Q TABLES E  ANCHE IL BEEP
            obs_dict = ({'x': obs[0][0], 'y': obs[0][1], 'beep': obs[1], 'color': color_idx},)



        action = [action]
        # print("observation", obs_dict)

        next_automata_states = [tg.step(obs_dict, action) for tg in self.temp_goals]
        #DFA.to_dot(self, "/home/alee8", next_automata_states)



        temp_goal_rewards = [
            tg.observe_reward(is_terminal_state=done)
            for tg in self.temp_goals
        ]
        total_goal_rewards = sum(temp_goal_rewards)
        ##print("\nobs:", obs_dict, "action:", action, "next_automata_states", next_automata_states, total_goal_rewards, "total_goal_rewards", temp_goal_rewards, "temp_goal_rewards\n")
        '''if (total_goal_rewards != 0.0):
            breakpoint()'''



        if any(r != 0.0 for r in temp_goal_rewards):
            logger.debug("Non-zero goal rewards: {}".format(temp_goal_rewards))
        # print("next_automata_states", next_automata_states)
        obs_prime = (obs_dict, next_automata_states)
        reward_prime = reward + total_goal_rewards
        '''if total_goal_rewards != 0.0:
            print(total_goal_rewards, "total_goal_rewards")
            breakpoint()'''

        ##print(reward_prime,"reward_prime", "=" , total_goal_rewards,"+", reward, "(total_goal_rewards + reward)")
        '''if total_goal_rewards != 0.0:
            breakpoint()'''
        '''if reward_prime > 1.0:
            print(reward_prime, "weee")
            breakpoint()
        else:
            print(reward_prime, "weeqqqqe")'''


        return obs_prime, reward_prime, done, info

    def reset(self, **_kwargs):
        """Reset the Gym environment."""
        #obs = ({'x': 1, 'y': 1, 'beep': False, 'color': 0},)
        #obs = super().reset()
        #print("obs_dict-obs_dictobs_dictobs_dictobs_dictobs_dictobs_dictobs_dict", obs)
        #print("ciaooooo")
        for tg in self.temp_goals:
            tg.reset()
        #breakpoint()
        automata_states = [tg.reset() for tg in self.temp_goals]
        #return obs, automata_states
        #print("automata_states", automata_states)

        #breakpoint()
        return automata_states


