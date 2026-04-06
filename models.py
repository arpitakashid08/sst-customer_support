# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Customer Support OpenEnv Environment.

This environment simulates customer support interactions.
"""

from typing import Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


IssueType = Literal["order", "refund", "payment", "complaint", "general"]
Priority = Literal["low", "medium", "high"]
ActionType = Literal[
    "reply_to_customer",
    "create_support_ticket",
    "escalate_to_human",
]


class MyAction(Action):
    """Action for the customer support environment."""

    action: ActionType = Field(
        ...,
        description="Action to take for the customer",
    )
    response: str | None = Field(
        default=None,
        description="Optional response text returned to the customer",
    )


class MyObservation(Observation):
    """Observation from the customer support environment."""

    customer_message: str = Field(
        default="",
        description="The customer's latest message",
    )
    issue_type: IssueType = Field(
        default="general",
        description="Detected issue type for the customer request",
    )
    priority: Priority = Field(
        default="medium",
        description="Priority level for the issue",
    )
