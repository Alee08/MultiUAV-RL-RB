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

"""This module contains core library functions and classes."""

import random
from abc import ABC, abstractmethod
from typing import Any, Collection, List, Optional, Sequence, cast

import gym
import numpy as np

from yarllib.helpers.base import assert_
from yarllib.helpers.history import AgentObs, EpisodeAgentObs, History
from yarllib.types import AgentObservation, State


class LearningEventListener(ABC):
    """
    The learning event listener interface.

    Each event listener is associated to the learning context.
    It can listen to the following events:
    - on_session_begin
    - on_session_end
    - on_episode_begin
    - on_episode_end
    - on_step_begin
    - on_step_end
    """

    _context: Optional["Context"] = None

    @property
    def context(self) -> "Context":
        """Get the context."""
        assert_(self._context is not None, "Context not set.")
        return cast(Context, self._context)

    @context.setter
    def context(self, value: Optional["Context"] = None) -> None:
        """
        Set the context.

        This method is called from the context class at the
        beginning of each session, in order to bind
        the listeners to the same context,
        and and the end of each session, to unset it.

        :param value: the reference to the context.
        :return: None
        """
        self._context = value

    def on_session_begin(self, *args, **kwargs) -> None:
        """On session begin event."""

    def on_session_end(self, exception: Optional[Exception], *args, **kwargs) -> None:
        """On session end event."""

    def on_episode_begin(self, episode, **kwargs) -> None:
        """On episode begin event."""

    def on_episode_end(self, episode, **kwargs) -> None:
        """On episode end event."""

    def on_step_begin(self, step, action, **kwargs) -> None:
        """On step begin event."""

    def on_step_end(self, step, agent_observation: AgentObservation, **kwargs) -> None:
        """On step end event."""


class HistoryCallback(LearningEventListener):
    """Record an entire history of a simulation with the environment."""

    def __init__(self):
        """Initialize the history callback."""
        super().__init__()
        self.current_episode: List[AgentObs] = []
        self.episodes: List[EpisodeAgentObs] = []

    def get_history(self) -> History:
        """Get the history."""
        is_training = self.context.is_training
        seed = self.context.seed
        name = self.context.experiment_name
        return History(self.episodes, is_training=is_training, seed=seed, name=name)

    def on_session_begin(self, *args, **kwargs) -> None:
        """On session begin event."""
        self.current_episode = []
        self.episodes = []

    def on_step_end(self, step, agent_observation: AgentObservation, **kwargs) -> None:
        """On step end event."""
        s, a, r, sp = agent_observation[:-1]
        self.current_episode.append((s, a, r, sp))

    def on_episode_end(self, episode, **kwargs) -> None:
        """On episode end event."""
        self.episodes.append(self.current_episode)
        self.current_episode = []

    def on_session_end(self, *args, **kwargs) -> None:
        """On session end event."""
        # the training might have been stopped
        # in the middle of an episode
        if len(self.current_episode) > 0:
            self.episodes.append(self.current_episode)
            self.current_episode = []


class Model(LearningEventListener):
    """An RL Model."""

    @abstractmethod
    def get_best_action(self, state: State) -> Any:
        """Get the best action for the current model."""

    def agent(self) -> "Agent":
        """Wrap the model with an agent."""
        return Agent(self)

    def get_action(self, current_state: State) -> Any:
        """
        Get the action.

        It forwards the call to the current active policy.
        """
        # by default, we follow the policy of the learning context.
        return self.context.policy.get_action(current_state)

    def freeze(self) -> "Model":
        """
        Freeze the model. Useful for testing.

        If this doesn't make sense for your model, then
        override the concrete class and handle the behaviour
        of the model whether it is in "training" mode or not.
        """
        return FreezedModel(self)


class FreezedModel(Model):
    """This class makes a model suitable for testing."""

    def __init__(self, model: Model):
        """
        Initialize a freezed model.

        :param model: the model to wrap.
        """
        self._model = model

    @property
    def context(self) -> "Context":
        """Get the context."""
        return self._model.context

    @context.setter
    def context(self, value: Optional["Context"] = None) -> None:
        """Set the context."""
        self._model._context = value

    def get_best_action(self, state: State) -> None:
        """Get the best action."""
        return self._model.get_best_action(state)

    def get_action(self, state: State) -> Any:
        """Get the action."""
        return self._model.get_action(state)


class Policy(LearningEventListener):
    """A policy."""

    _model: Optional[Model] = None
    _action_space: Optional[gym.spaces.Space] = None

    @property
    def action_space(self) -> gym.spaces.Space:
        """Get the action space."""
        assert_(self._model is not None, "Action space is not set.")
        return self._action_space

    @action_space.setter
    def action_space(self, value: Optional[gym.spaces.Space]) -> None:
        """Get the action space."""
        self._action_space = value

    @abstractmethod
    def get_action(self, state: State) -> Any:
        """Get the action."""

    @property
    def model(self) -> Model:
        """Get the context."""
        assert_(self._model is not None, "Model not set.")
        return cast(Model, self._model)

    @model.setter
    def model(self, value: Optional[Model] = None) -> None:
        """
        Set the model`.

        This method is called from the context class at the
        beginning of each session, in order to bind
        the listeners to the same context,
        and and the end of each session, to unset it.

        :param value: the reference to the model.
        :return: None
        """
        self._model = value


class Context:
    """Training/Testing context."""

    def __init__(
        self,
        environment: gym.Env,
        model: Model,
        policy: Policy,
        nb_episodes: Optional[int],
        nb_steps: Optional[int],
        listeners: Collection[LearningEventListener],
        is_training: bool,
        seed: Optional[int] = None,
        experiment_name: str = "",
    ):
        """Initialize the context."""
        self.model = model
        self.environment = environment
        self.policy = policy
        self.history_callback = HistoryCallback()
        self.listeners = list(listeners) + [self.history_callback]
        self.is_training = is_training
        self.seed = seed
        self.experiment_name = experiment_name
        self.nb_episodes = nb_episodes
        self.nb_steps = nb_steps
        self.current_step = 0
        self.current_episode_step = 0
        self.current_episode = 0

    def get_history(self) -> History:
        """Get the history."""
        return self.history_callback.get_history()

    def begin_session(self) -> None:
        """Trigger the begin session event."""
        if self.seed is not None:
            self._set_seed()
        self.policy.model = self.model
        self.policy.action_space = self.environment.action_space
        for listener in self.listeners:
            listener.context = self
            listener.on_session_begin()

    def end_session(self, exception: Optional[Exception] = None) -> None:
        """Trigger the end session event."""
        for listener in self.listeners:
            listener.on_session_end(exception)
            listener._context = None
        self.policy._model = None
        self.policy._action_space = None
        if exception is not None:
            raise exception

    def begin_episode(self) -> None:
        """Trigger the begin episode event."""
        for listener in self.listeners:
            listener.on_episode_begin(self.current_episode)

    def end_episode(self) -> None:
        """Trigger the end episode event."""
        for listener in self.listeners:
            listener.on_episode_end(self.current_episode)
        self.current_episode_step = 0
        self.current_episode += 1

    def begin_step(self, action) -> None:
        """Trigger the begin step event."""
        for listener in self.listeners:
            listener.on_step_begin(self.current_episode_step, action)

    def end_step(self, agent_observation: AgentObservation) -> None:
        """Trigger the end step event."""
        for listener in self.listeners:
            listener.on_step_end(self.current_episode_step, agent_observation)
        self.current_episode_step += 1
        self.current_step += 1

    def _set_seed(self):
        """Set the random seed."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.environment.seed(self.seed)
        self.environment.observation_space.seed(self.seed)
        self.environment.action_space.seed(self.seed)

    def is_session_done(self) -> bool:
        """
        Check whether the session is done.

        That is: either we run out of episodes or run out of steps.
        """
        assert self.nb_episodes is not None or self.nb_steps is not None
        is_beyond_max_episode = (
            self.nb_episodes is not None and self.current_episode >= self.nb_episodes
        )
        is_beyond_max_step = (
            self.nb_steps is not None and self.current_step >= self.nb_steps
        )
        return is_beyond_max_episode or is_beyond_max_step


class Agent(ABC):
    """
    The agent class.

    This is where the main training/testing loop is implemented.
    This class acts as an orchestrator between all the components
    during the training and testing of the model.
    """

    def __init__(self, model: Model):
        """
        Initialize an agent.

        :param model: the model of the learning agent.
        """
        self.model = model

    def _play(
        self,
        env: gym.Env,
        policy: Policy,
        nb_episodes: Optional[int] = None,
        nb_steps: Optional[int] = None,
        callbacks: Sequence[LearningEventListener] = (),
        is_training: bool = True,
        seed: Optional[int] = None,
        experiment_name: str = "",
    ) -> History:
        """
        Run a training/testing session of the agent.

        :param env: the environment to learn from.
        :param nb_steps: the number of steps.
        :return: None.
        """
        context = self._make_context(
            env,
            policy,
            nb_episodes,
            nb_steps,
            callbacks,
            is_training,
            seed,
            experiment_name,
        )
        context.begin_session()
        done = False
        current_state = env.reset()
        context.begin_episode()
        exception = None
        try:
            while not context.is_session_done():
                if done:
                    context.end_episode()
                    current_state = env.reset()
                    done = False
                    if not context.is_session_done():
                        context.begin_episode()
                    continue
                action = context.model.get_action(current_state)
                context.begin_step(action)
                next_state, reward, done, _ = env.step(action)
                context.end_step((current_state, action, reward, next_state, done))
                current_state = next_state
        except KeyboardInterrupt:
            pass
        except Exception as e:
            exception = e
        history = context.get_history()
        context.end_session(exception)
        return history

    def train(self, *args, **kwargs) -> History:
        """Train the agent."""
        assert_("is_training" not in kwargs, "Cannot specify the 'is_training' flag.")
        return self._play(*args, is_training=True, **kwargs)  # type: ignore

    def test(self, *args, **kwargs) -> History:
        """Test the agent."""
        assert_("is_training" not in kwargs, "Cannot specify the 'is_training' flag.")
        return self._play(*args, is_training=False, **kwargs)  # type: ignore

    def _make_context(
        self,
        env: gym.Env,
        policy: Policy,
        nb_episodes: Optional[int] = None,
        nb_steps: Optional[int] = None,
        callbacks: Sequence[LearningEventListener] = (),
        is_training: bool = True,
        seed: Optional[int] = None,
        experiment_name: str = "",
    ):
        """Make the context."""
        # the model is a listener only if we are training.
        model = self.model.freeze() if not is_training else self.model
        return Context(
            env,
            model,
            policy,
            nb_episodes,
            nb_steps,
            [policy, model, *callbacks],
            is_training,
            seed=seed,
            experiment_name=experiment_name,
        )
