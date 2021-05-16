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

"""Test the Q-Learning and Sarsa implementation using FrozenLake."""

import pytest
from gym.envs.toy_text import FrozenLakeEnv
from gym.wrappers import TimeLimit

from yarllib.models.tabular import TabularQLearning, TabularSarsa
from yarllib.policies import EpsGreedyPolicy, GreedyPolicy


@pytest.mark.parametrize("model_class", [TabularQLearning, TabularSarsa])
@pytest.mark.parametrize("sparse", [True, False])
def test_frozenlake(model_class, sparse):
    """Test Q-Learning implementation on N-Chain environment."""
    env = FrozenLakeEnv(is_slippery=False)
    env = TimeLimit(env, max_episode_steps=env.observation_space.n * 5)
    agent = model_class(env.observation_space, env.action_space, gamma=0.9).agent()
    agent.train(env, policy=EpsGreedyPolicy(epsilon=0.5), nb_steps=75000)
    evaluation = agent.test(env, policy=GreedyPolicy(), nb_episodes=10)
    assert evaluation.total_rewards.mean() == 1.0
