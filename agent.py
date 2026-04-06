"""PPO-based agent for the customer support environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO

try:
    from support_env import ACTION_LABELS, encode_observation, scenario_by_id
except ImportError:
    from my_env.support_env import ACTION_LABELS, encode_observation, scenario_by_id


@dataclass(frozen=True)
class AgentResult:
    """Agent inference output."""

    action_id: int
    action_label: str
    action_probs: Dict[str, float]


class PPOAgent:
    """Loads a trained PPO model and runs inference."""

    def __init__(self, model_path: str = "support_agent"):
        self.model = PPO.load(model_path)
        self._expected_shape = self.model.observation_space.shape or (1,)

    def predict(self, observation: np.ndarray, deterministic: bool = False) -> AgentResult:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        obs = self._coerce_observation(obs)
        action_id, _ = self.model.predict(obs, deterministic=deterministic)
        action_id = int(np.asarray(action_id).item())
        action_probs = self._action_probabilities(observation)
        return AgentResult(
            action_id=action_id,
            action_label=ACTION_LABELS[action_id],
            action_probs=action_probs,
        )

    def _action_probabilities(self, observation: np.ndarray) -> Dict[str, float]:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        obs = self._coerce_observation(obs)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        dist = self.model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.detach().cpu().numpy()[0]
        return {label: float(probs[idx]) for idx, label in enumerate(ACTION_LABELS)}

    def _coerce_observation(self, obs: np.ndarray) -> np.ndarray:
        """Coerce observation to the shape expected by the loaded model."""
        expected_dim = int(self._expected_shape[0])
        current_dim = int(obs.shape[1])
        if current_dim == expected_dim:
            return obs
        if expected_dim == 1:
            return obs[:, :1]
        if current_dim < expected_dim:
            pad = np.zeros((obs.shape[0], expected_dim - current_dim), dtype=obs.dtype)
            return np.concatenate([obs, pad], axis=1)
        return obs[:, :expected_dim]


def decide_action_from_scenario(scenario_id: int) -> AgentResult:
    """Convenience wrapper for demo usage."""
    scenario = scenario_by_id(scenario_id)
    observation = encode_observation(scenario)
    agent = PPOAgent()
    return agent.predict(observation)
