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

"""Test the Q-Learning and Sarsa implementation using Cliff."""

import gym
import numpy as np
from gym.envs.toy_text import CliffWalkingEnv
from gym.wrappers import TimeLimit

from yarllib.experiment_utils import run_experiments
from yarllib.models.tabular import TabularQLearning, TabularSarsa
from yarllib.policies import EpsGreedyPolicy


class CliffWalkingEnvWrapper(gym.Wrapper):
    """A wrapper to the CliffWalking environment."""

    def step(self, action):
        """Stop the episode when the cliff is reached."""
        s, r, d, i = super().step(action)
        if r == -100:
            d = True
        return s, r, d, i


def test_cliff():
    """Test that Sarsa > QLearning in the Cliff Environment."""
    env = CliffWalkingEnv()
    env = CliffWalkingEnvWrapper(env)
    env = TimeLimit(env, max_episode_steps=50)

    def make_sarsa():
        return TabularSarsa(env.observation_space, env.action_space).agent()

    def make_qlearning():
        return TabularQLearning(env.observation_space, env.action_space).agent()

    nb_episodes = 500
    nb_runs = 5
    policy = EpsGreedyPolicy(0.1)

    _, sarsa_histories = run_experiments(
        make_sarsa, env, policy, nb_runs=nb_runs, nb_episodes=nb_episodes
    )
    _, qlearning_histories = run_experiments(
        make_qlearning, env, policy, nb_runs=nb_runs, nb_episodes=nb_episodes
    )

    assert len(sarsa_histories) == 5
    assert len(qlearning_histories) == 5

    sarsa_total_rewards = np.asarray([h.total_rewards for h in sarsa_histories])
    qlearning_total_rewards = np.asarray([h.total_rewards for h in qlearning_histories])

    sarsa_last_reward_avg = sarsa_total_rewards.mean(axis=0)[-100:].mean()
    qlearning_last_reward_avg = qlearning_total_rewards.mean(axis=0)[-100:].mean()

    # compare sarsa and q-learning on the averaged total reward in the last episode
    # sarsa is always better than q-learning
    assert sarsa_last_reward_avg > -30 > qlearning_last_reward_avg > -40
