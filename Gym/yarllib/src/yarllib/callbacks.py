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

"""This module contains callbacks to customize the training/testing loop."""
import logging
import shutil
from pathlib import Path

from PIL import Image

from yarllib.core import LearningEventListener
from yarllib.types import AgentObservation


class RenderEnv(LearningEventListener):
    """An OpenAI Gym renderer implemented as listener."""

    def on_episode_begin(self, *args, **kwargs) -> None:
        """On episode begin event."""
        self.context.environment.render()

    def on_episode_end(self, *args, **kwargs) -> None:
        """On episode begin event."""
        self.context.environment.render()

    def on_step_end(self, *args, **kwargs) -> None:
        """On step end event."""
        self.context.environment.render()


class LoggingCallback(LearningEventListener):
    """Callback for logging purposes."""

    def __init__(
        self, logger_name: str, level: int = logging.INFO, log_interval: int = 100
    ):
        """
        Initialize the logger callback.

        :param logger_name: the logger name.
        :param level: the logging level.
        :param log_interval: the episode interval when to log messages.
        """
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level)
        self.log_interval = log_interval

    def on_session_begin(self, *args, **kwargs) -> None:
        """On session begin event."""
        self.logger.info("on_session_begin", **kwargs)

    def on_session_end(self, *args, **kwargs) -> None:
        """On session end event."""
        self.logger.info("on_session_end", **kwargs)

    def on_episode_begin(self, episode, *args, **kwargs) -> None:
        """On episode begin event."""
        if episode % self.log_interval == 0:
            self.logger.info(f"on_episode_begin: episode={episode}", **kwargs)

    def on_episode_end(self, episode, **kwargs) -> None:
        """On episode end event."""
        if episode % self.log_interval == 0:
            self.logger.info(f"on_episode_end: episode={episode}", **kwargs)

    def on_step_begin(self, step, action, **kwargs) -> None:
        """On step begin event."""
        self.logger.debug(f"on_step_begin: step={step}, action={action}", **kwargs)

    def on_step_end(self, step, agent_observation: AgentObservation, **kwargs) -> None:
        """On step end event."""
        self.logger.debug(
            f"on_step_end: step={step}, agent_observation={agent_observation}", **kwargs
        )


class FrameCapture(LearningEventListener):
    """Capture frames from the game."""

    def __init__(self, dest_dir: Path):
        """
        Initialize the callback.

        :param dest_dir: the destination directory.
        """
        self.dest_dir = Path(dest_dir)
        if self.dest_dir.exists():
            shutil.rmtree(self.dest_dir)
        self.dest_dir.mkdir()

    def save_frame(self) -> None:
        """
        Save the frame.

        :return: None
        """
        rgb_array = self.context.environment.render("rgb_array")
        img = Image.fromarray(rgb_array)
        step = self.context.current_episode_step
        filename = "{:010}.jpeg".format(step)
        img.save(self.dest_dir / str(self.context.current_episode) / filename)

    def on_episode_begin(self, episode, **kwargs) -> None:
        """Handle the episode begin."""
        (self.dest_dir / str(episode)).mkdir(parents=True)

    def on_episode_end(self, episode, **kwargs) -> None:
        """Handle the episode end."""
        self.save_frame()

    def on_step_begin(self, step, action, **kwargs) -> None:
        """Handle the step begin."""
        self.save_frame()
