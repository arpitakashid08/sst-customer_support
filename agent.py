"""Hybrid agent: OpenAI if available, PPO fallback otherwise."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import os

import numpy as np
from openai import OpenAI
from stable_baselines3 import PPO

try:
    from support_env import (
        ACTION_LABELS,
        decode_observation,
        default_action_for_issue,
    )
except ImportError:
    from my_env.support_env import (
        ACTION_LABELS,
        decode_observation,
        default_action_for_issue,
    )


@dataclass(frozen=True)
class AgentResult:
    """Agent inference output."""

    action_id: int
    action_label: str
    action_probs: Dict[str, float]


class SupportAgent:
    """Uses OpenAI to choose an action, with rule-based fallback."""

    def __init__(self, client: OpenAI | None, model_name: str, ppo_path: str = "support_agent"):
        self.client = client
        self.model_name = model_name
        self.use_ppo_only = os.getenv("USE_PPO_ONLY", "0").lower() in {"1", "true", "yes"}
        self.ppo = None
        self._ppo_action_labels: list[str] = []
        self._ppo_expected_shape: tuple[int, ...] | None = None
        try:
            self.ppo = PPO.load(ppo_path)
            n_actions = int(self.ppo.action_space.n)  # type: ignore[attr-defined]
            self._ppo_action_labels = ACTION_LABELS[:n_actions]
            self._ppo_expected_shape = self.ppo.observation_space.shape  # type: ignore[attr-defined]
        except Exception:
            self.ppo = None

    def act(self, observation: np.ndarray) -> AgentResult:
        scenario = decode_observation(observation)
        if self.use_ppo_only:
            return self._ppo_or_rule(scenario.issue_type, observation)
        prompt = (
            "You are a customer support agent. "
            "Choose exactly one action from: reply, escalate, create_ticket, request_info.\n"
            f"Customer message: {scenario.message}\n"
            f"Issue type: {scenario.issue_type}\n"
            f"Priority: {scenario.priority}\n"
            "Return ONLY the action."
        )

        if self.client is None:
            return self._ppo_or_rule(scenario.issue_type, observation)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
            )
            raw_text = response.choices[0].message.content.strip()
            action_label = self._parse_action(raw_text.lower(), scenario.issue_type)
            return AgentResult(
                action_id=ACTION_LABELS.index(action_label),
                action_label=action_label,
                action_probs={"response_text": raw_text},
            )
        except Exception:
            return self._ppo_or_rule(scenario.issue_type, observation)

        action_id = ACTION_LABELS.index(action_label)
        return AgentResult(action_id=action_id, action_label=action_label, action_probs={})

    def _parse_action(self, text: str, issue_type: str) -> str:
        if "create_ticket" in text or "create ticket" in text:
            return "create_support_ticket"
        if "escalate" in text:
            return "escalate_to_human"
        if "request_info" in text or "request info" in text:
            return "request_info"
        if "reply" in text:
            return "reply_to_customer"
        return default_action_for_issue(issue_type)

    def _ppo_or_rule(self, issue_type: str, observation: np.ndarray) -> AgentResult:
        if self.ppo is not None and self._ppo_action_labels:
            obs = np.asarray(observation, dtype=np.float32)
            if obs.ndim == 1:
                obs = obs.reshape(1, -1)
            obs = self._coerce_ppo_observation(obs)
            action_id, _ = self.ppo.predict(obs, deterministic=False)
            action_id = int(np.asarray(action_id).item())
            action_label = self._ppo_action_labels[action_id]
            return AgentResult(action_id=ACTION_LABELS.index(action_label), action_label=action_label, action_probs={})

        action_label = default_action_for_issue(issue_type)
        action_id = ACTION_LABELS.index(action_label)
        return AgentResult(action_id=action_id, action_label=action_label, action_probs={})

    def _coerce_ppo_observation(self, obs: np.ndarray) -> np.ndarray:
        """Match the PPO model's expected observation shape."""
        if not self._ppo_expected_shape:
            return obs
        expected_dim = int(self._ppo_expected_shape[0])
        current_dim = int(obs.shape[1])
        if current_dim == expected_dim:
            return obs
        if expected_dim == 1:
            return obs[:, :1]
        if current_dim < expected_dim:
            pad = np.zeros((obs.shape[0], expected_dim - current_dim), dtype=obs.dtype)
            return np.concatenate([obs, pad], axis=1)
        return obs[:, :expected_dim]


__all__ = ["SupportAgent", "AgentResult"]
