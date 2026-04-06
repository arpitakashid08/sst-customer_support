---
title: Customer Support AI Environment
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_file: server/app.py
pinned: false
app_port: 8000
---

# Customer Support AI Environment (RL + OpenEnv)

A reinforcement‑learning customer support environment built for the OpenEnv Hackathon. The agent is trained with PPO (Stable‑Baselines3) and serves actions through the OpenEnv API and a Gradio demo UI.

## Project Overview

This project simulates customer support scenarios (order issues, refunds, payments, complaints, general queries). The environment provides a structured observation, the PPO agent selects an action, and rewards are computed based on correctness.

## Reinforcement Learning Approach

- **Algorithm:** PPO (Stable‑Baselines3)
- **Policy:** `MlpPolicy`
- **Environment:** one‑step Gymnasium environment (`support_env.py`)
- **Model:** `support_agent.zip`

## Environment Design

### Observation Space
A 3‑dimensional continuous vector:

1. `scenario_id` normalized to `[0, 1]`
2. `priority` normalized to `[0, 1]`
3. constant `1.0` (bias feature)

`observation_space = Box(low=0.0, high=1.0, shape=(3,), dtype=float32)`

### Action Space
Discrete actions:

- `0` → `reply_to_customer`
- `1` → `create_support_ticket`
- `2` → `escalate_to_human`

`action_space = Discrete(3)`

### Reward Function
Positive reward for correct handling, negative penalty for wrong actions:

- `order` → `create_support_ticket` → `+5`
- `refund` → `create_support_ticket` → `+10`
- `payment` → `create_support_ticket` → `+10`
- `complaint` → `escalate_to_human` → `+10`
- `general` → `reply_to_customer` → `+2`
- Incorrect actions → `-1`

## Training

```bash
python3 train_agent.py
```

Model is saved to:
```
support_agent.zip
```

You can configure training via env vars:

```bash
PPO_TIMESTEPS=20000 PPO_SEED=42 python3 train_agent.py
```

## Gradio Demo

```bash
python3 -m gradio_demo.demo
```

The UI displays:
- Detected scenario
- PPO action
- Reward received
- Reward progression graph
- PPO action probabilities

## OpenEnv Workflow

### Test Locally
```bash
uv sync
uv run server
```
Open:
```
http://127.0.0.1:8000/
```

### Deploy
```bash
openenv push --repo-id <username>/<repo>
```

If you lack direct push access:
```bash
openenv push --repo-id <username>/<repo> --create-pr
```
Then merge the PR and Factory Reboot the Space.

## Files

- `support_env.py` – Gymnasium environment
- `agent.py` – PPO inference wrapper
- `train_agent.py` – PPO training script
- `server/` – OpenEnv server (FastAPI)
- `gradio_demo/` – Demo UI

---

This project is fully compatible with Hugging Face Spaces (Docker) and the OpenEnv Hackathon submission workflow.
