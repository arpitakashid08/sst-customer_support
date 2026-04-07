# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Customer Support Environment Implementation.

Simulates customer support queries and evaluates actions for rewards.
"""

from __future__ import annotations

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import MyAction, MyObservation
    from support_env import ACTION_LABELS, SCENARIOS, reward_for_action, scenario_by_id
except ImportError:
    from my_env.models import MyAction, MyObservation
    from my_env.support_env import ACTION_LABELS, SCENARIOS, reward_for_action, scenario_by_id


CASE_LIBRARY = [scenario for scenario in SCENARIOS]


class MyEnvironment(Environment):
    """
    Customer support environment with reward evaluation.
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the customer support environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._rng = random.Random()
        self._current_case = None

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: object,
    ) -> MyObservation:
        """
        Reset the environment.

        Returns:
            MyObservation with a new customer message
        """
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._reset_count += 1
        if seed is not None:
            self._rng.seed(seed)
        self._current_case = self._rng.choice(CASE_LIBRARY)

        return MyObservation(
            customer_message=self._current_case.message,
            issue_type=self._current_case.issue_type,
            priority=self._current_case.priority,
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: MyAction,
        timeout_s: float | None = None,
        **kwargs: object,
    ) -> MyObservation:  # type: ignore[override]
        """
        Execute a step in the environment by evaluating the action.

        Args:
            action: MyAction containing the chosen action

        Returns:
            MyObservation with reward and done flag
        """
        self._state.step_count += 1

        if self._current_case is None:
            self._current_case = self._rng.choice(CASE_LIBRARY)

        try:
            action_id = ACTION_LABELS.index(action.action)
        except ValueError:
            action_id = 0

        reward = reward_for_action(self._current_case, action_id)
        correct_action = ACTION_LABELS[self._current_case.correct_action]
        is_correct = action_id == self._current_case.correct_action

        return MyObservation(
            customer_message=self._current_case.message,
            issue_type=self._current_case.issue_type,
            priority=self._current_case.priority,
            done=True,
            reward=reward,
            metadata={
                "scenario": self._current_case.name,
                "action_taken": action.action,
                "correct_action": correct_action,
                "is_correct": is_correct,
                "step": self._state.step_count,
            },
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
