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

"""Helpers to plot experiment results."""

import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import DataFrame

from yarllib.helpers.history import History


def plot_summaries(
    history_lists: Sequence[Sequence[History]], labels: Sequence[str], prefix: str = "."
):
    """
    Plot summaries of several experiments.

    :param history_lists: the list of experiments, i.e. list of list of histories.
    :param labels: the labels associated to each experiment.
    :param prefix: path prefix where to save the plots.
    :return: None
    """
    assert len(history_lists) == len(
        labels
    ), "Please provide the correct number of labels."
    attributes = ["total_rewards", "average_rewards", "lengths"]
    figures = []
    for attribute in attributes:
        f = plt.figure()
        ax = f.add_subplot()
        ax.set_title(attribute)
        for label, history_list in zip(labels, history_lists):
            _plot_history_attribute(history_list, attribute, label, ax=ax)
        figures.append(f)
        plt.savefig(os.path.join(prefix, attribute + ".svg"))
    return figures


def _plot_history_attribute(
    histories: Sequence[History], attribute: str, label, ax=None
):
    """Plot a certain attribute of a collection of histories."""
    data = np.asarray([getattr(h, attribute) for h in histories])
    df = DataFrame(data.T)

    df_mean = df.mean(axis=1)
    df_std = df.std(axis=1)
    sns_ax = sns.lineplot(df_mean.index, df_mean, label=label, ax=ax)
    sns_ax.fill_between(df_mean.index, df_mean - df_std, df_mean + df_std, alpha=0.3)
