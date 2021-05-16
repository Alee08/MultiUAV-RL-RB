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

"""This module includes utilities to run many experiments."""

import multiprocessing
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple

import gym

from yarllib.core import Agent, LearningEventListener, Policy
from yarllib.helpers.base import assert_
from yarllib.helpers.history import History


def _do_job(agent, env, policy, seed, nb_episodes, callbacks, experiment_name):
    """Run the agent training."""
    history = agent.train(
        env,
        policy,
        nb_episodes=nb_episodes,
        seed=seed,
        callbacks=callbacks,
        experiment_name=experiment_name,
    )
    return agent, history


def _raise_exception(e):
    raise e


def _seed_to_str(max_seed: int, seed: int) -> str:
    """Transform a seed into string."""
    nb_digits = len(str(max_seed))
    return "{{:0{}}}".format(nb_digits).format(seed)


def run_experiments(
    make_agent: Callable,
    env: gym.Env,
    policy: Policy,
    nb_runs: int = 50,
    nb_episodes: int = 500,
    nb_workers: int = 8,
    seeds: Optional[Sequence[int]] = None,
    callbacks: Sequence[LearningEventListener] = (),
    name_prefix: str = "experiment",
) -> Tuple[List[Agent], List[History]]:
    """
    Run many experiments with multiprocessing.

    :param make_agent: a callable to make an agent.
    :param env: the environment to use.
    :param policy: the policy.
    :param nb_runs: the number of runs.
    :param nb_episodes: the number of episodes.
    :param nb_workers: the number of workers.
    :param seeds: a list of seeds; if None, the range [0, nb_runs-1] is used.
    :param callbacks: a list callbacks.
    :param name_prefix: the prefix to each experiment.
    :return: a list of histories, one for each run.
    """
    agents = []
    histories = []

    if seeds is None:
        seeds = list(range(0, nb_runs))
    assert_(
        len(seeds) == nb_runs,
        f"The number of seeds {len(seeds)} is different from the number of runs {nb_runs}.",
    )
    agent = make_agent()
    pool = multiprocessing.Pool(processes=nb_workers)
    _current_seed_to_str = partial(_seed_to_str, max(seeds))
    results = [
        pool.apply_async(
            _do_job,
            args=(
                agent,
                env,
                policy,
                seed,
                nb_episodes,
                callbacks,
                name_prefix + "-" + _current_seed_to_str(seed),
            ),
            error_callback=_raise_exception,
        )
        for seed in seeds
    ]
    try:
        for p in results:
            p.wait()
    except KeyboardInterrupt:
        pass

    for p in filter(lambda x: x.ready(), results):
        agent, history = p.get()
        agents.append(agent)
        histories.append(history)
    return agents, histories
