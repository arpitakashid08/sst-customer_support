"""Run a single inference episode with structured logging."""

from __future__ import annotations

import os

from openai import OpenAI

from agent import SupportAgent
from support_env import ACTION_NAME_TO_ID, CustomerSupportEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None


def main() -> None:
    print("START", flush=True)

    env = CustomerSupportEnv()
    agent = SupportAgent(client, MODEL_NAME)

    obs, _ = env.reset()
    done = False
    step = 0

    while not done:
        result = agent.act(obs)
        action_id = ACTION_NAME_TO_ID[result.action_label]
        obs, reward, terminated, truncated, info = env.step(action_id)
        done = terminated or truncated

        print(f"STEP: {step}", flush=True)
        print(f"OBSERVATION: {obs}", flush=True)
        print(f"ACTION: {result.action_label}", flush=True)
        print(f"REWARD: {reward}", flush=True)

        step += 1

    print("END", flush=True)


if __name__ == "__main__":
    main()
