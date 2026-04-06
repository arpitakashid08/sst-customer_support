"""Train a PPO agent for the customer support environment."""

from __future__ import annotations

import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from support_env import CustomerSupportEnv


def train(total_timesteps: int = 20_000, seed: int = 42) -> None:
    """Train PPO and save the model."""
    env = DummyVecEnv([lambda: CustomerSupportEnv(seed=seed)])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        n_steps=256,
        batch_size=256,
    )
    model.learn(total_timesteps=total_timesteps)
    model.save("support_agent")
    print("Training complete. Model saved as support_agent.zip")


if __name__ == "__main__":
    timesteps = int(os.getenv("PPO_TIMESTEPS", "20000"))
    seed = int(os.getenv("PPO_SEED", "42"))
    train(total_timesteps=timesteps, seed=seed)
