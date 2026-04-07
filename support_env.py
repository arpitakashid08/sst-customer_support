"""Gymnasium environment for customer support RL training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


ACTION_LABELS: List[str] = [
    "reply_to_customer",
    "create_support_ticket",
    "escalate_to_human",
    "request_info",
]

PRIORITY_TO_VALUE: Dict[str, float] = {"low": 0.0, "medium": 0.5, "high": 1.0}


@dataclass(frozen=True)
class Scenario:
    """Represents a customer support scenario."""

    scenario_id: int
    name: str
    message: str
    issue_type: str
    priority: str
    correct_action: int
    reward: float


SCENARIOS: List[Scenario] = [
    Scenario(
        scenario_id=0,
        name="Order Issue",
        message="My order hasn't arrived.",
        issue_type="order",
        priority="medium",
        correct_action=1,
        reward=5.0,
    ),
    Scenario(
        scenario_id=1,
        name="Refund Request",
        message="I want a refund for the last purchase.",
        issue_type="refund",
        priority="high",
        correct_action=1,
        reward=10.0,
    ),
    Scenario(
        scenario_id=2,
        name="Payment Issue",
        message="My payment failed but the money was deducted.",
        issue_type="payment",
        priority="high",
        correct_action=1,
        reward=10.0,
    ),
    Scenario(
        scenario_id=3,
        name="Complaint",
        message="This is unacceptable. I want to complain now.",
        issue_type="complaint",
        priority="high",
        correct_action=2,
        reward=10.0,
    ),
    Scenario(
        scenario_id=4,
        name="General Query",
        message="What are your business hours?",
        issue_type="general",
        priority="low",
        correct_action=3,
        reward=2.0,
    ),
]


def scenario_by_id(scenario_id: int) -> Scenario:
    return SCENARIOS[int(scenario_id)]

def decode_observation(observation: np.ndarray) -> Scenario:
    """Decode observation vector into the closest scenario."""
    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    scenario_value = float(obs[0])
    scenario_id = int(round(scenario_value * (len(SCENARIOS) - 1)))
    scenario_id = max(0, min(len(SCENARIOS) - 1, scenario_id))
    return scenario_by_id(scenario_id)


def encode_observation(scenario: Scenario) -> np.ndarray:
    """Encode a scenario into a numeric observation vector."""
    scenario_value = scenario.scenario_id / (len(SCENARIOS) - 1)
    priority_value = PRIORITY_TO_VALUE.get(scenario.priority, 0.5)
    return np.array([scenario_value, priority_value, 1.0], dtype=np.float32)


def reward_for_action(scenario: Scenario, action: int) -> float:
    """Reward function with positive/negative outcomes."""
    if action == scenario.correct_action:
        return scenario.reward
    return -1.0


def default_action_for_issue(issue_type: str) -> str:
    """Rule-based fallback action."""
    if issue_type in ("order", "refund", "payment"):
        return "create_support_ticket"
    if issue_type == "complaint":
        return "escalate_to_human"
    return "request_info"


ACTION_NAME_TO_ID = {label: idx for idx, label in enumerate(ACTION_LABELS)}


def infer_scenario_from_message(message: str) -> Scenario:
    """Lightweight scenario detection for the UI (not used for action selection)."""
    text = message.lower()
    if "refund" in text or "money back" in text:
        return SCENARIOS[1]
    if "payment" in text or "charged" in text or "card" in text:
        return SCENARIOS[2]
    if "complain" in text or "unacceptable" in text or "angry" in text:
        return SCENARIOS[3]
    if "order" in text or "delivery" in text or "shipment" in text:
        return SCENARIOS[0]
    return SCENARIOS[4]


class CustomerSupportEnv(gym.Env):
    """One-step customer support environment for PPO training."""

    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None):
        super().__init__()
        self._rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(len(ACTION_LABELS))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )
        self.current_scenario: Scenario | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.current_scenario = self._rng.choice(SCENARIOS)
        observation = encode_observation(self.current_scenario)
        info = {
            "scenario": self.current_scenario.name,
            "issue_type": self.current_scenario.issue_type,
            "priority": self.current_scenario.priority,
        }
        return observation, info

    def step(self, action: int):
        if self.current_scenario is None:
            self.current_scenario = self._rng.choice(SCENARIOS)

        reward = reward_for_action(self.current_scenario, int(action))
        terminated = True
        truncated = False
        observation = encode_observation(self.current_scenario)
        info = {
            "scenario": self.current_scenario.name,
            "issue_type": self.current_scenario.issue_type,
            "priority": self.current_scenario.priority,
            "action": ACTION_LABELS[int(action)],
            "reward": reward,
            "correct_action": ACTION_LABELS[self.current_scenario.correct_action],
        }
        return observation, reward, terminated, truncated, info
