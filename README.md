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

# Customer Support AI Environment (OpenEnv Hackathon)

A customer‑support environment built for the Meta PyTorch OpenEnv Hackathon.
It supports **OpenEnv API** (`reset/step/state`), a **Gymnasium RL environment**, and a **Gradio UI**.

This project uses **OpenAI for action selection** when an API key is present and falls back to **PPO or rule‑based logic** when it is not.

---

## Components and What They Do

### 1) `support_env.py` — Gymnasium Environment
- **Defines scenarios** and reward logic.
- **Observation space:** 3‑dim float vector
- **Action space:** Discrete(4)
- **Rewards:** positive for correct action, negative for incorrect.
- Provides helpers:
  - `encode_observation(scenario)`
  - `decode_observation(observation)`
  - `reward_for_action(scenario, action)`
  - `default_action_for_issue(issue_type)`

### 2) `agent.py` — Hybrid Agent (OpenAI + PPO + Rule fallback)
- **Primary:** OpenAI calls
- **Fallback:** PPO model (`support_agent.zip`)
- **Last fallback:** rule‑based
- Uses environment variables to configure OpenAI.

### 3) `inference.py` — Required Hackathon Script
- Uses **OpenAI client** (per spec)
- Reads env vars exactly:
  ```bash
  API_BASE_URL, MODEL_NAME, HF_TOKEN, LOCAL_IMAGE_NAME
  ```
- Runs one episode and prints **structured logs**:
  ```
  START
  STEP: <n>
  OBSERVATION: <obs>
  ACTION: <action>
  REWARD: <reward>
  END
  ```

### 4) `server/` — OpenEnv API (FastAPI)
- Exposes `reset`, `step`, `state`, and schema endpoints.
- Mounts Gradio UI at `/`.

### 5) `gradio_demo/` — UI
- Shows:
  - Scenario
  - Agent response
  - Action taken
  - Reward
  - Reward graph
  - OpenAI response text (when available)
- Uses the hybrid agent.

### 6) `train_agent.py`
- Trains a PPO agent (Stable‑Baselines3)
- Saves `support_agent.zip`

---

## OpenAI Usage

OpenAI is used in **two places**:

### ✅ `inference.py`
```python
from openai import OpenAI
client = OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY)
```

### ✅ `agent.py`
Uses OpenAI to produce an action label:

Actions allowed:
- `reply`
- `escalate`
- `create_ticket`
- `request_info`

If OpenAI fails (no key, error, etc), it falls back to PPO, then rule-based.

---

## Environment Variables

```bash
export OPENAI_API_KEY=your_key
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=...        # optional
export LOCAL_IMAGE_NAME=... # optional
```

---

## Actions

Action IDs:
- 0 → `reply_to_customer`
- 1 → `create_support_ticket`
- 2 → `escalate_to_human`
- 3 → `request_info`

---

## Reward Function

- `order` → `create_support_ticket` → +5
- `refund` → `create_support_ticket` → +10
- `payment` → `create_support_ticket` → +10
- `complaint` → `escalate_to_human` → +10
- `general` → `request_info` → +2
- incorrect action → -1

---

## Run Locally

```bash
cd /Users/arpitakashidgmail.com/Desktop/sst/my_env
uv sync
uv run server
```

Open:
```
http://127.0.0.1:8000/
```

---

## Run Inference (Hackathon Required)

```bash
cd /Users/arpitakashidgmail.com/Desktop/sst/my_env
python3 inference.py
```

Output:
```
START
STEP: 0
OBSERVATION: [0.5 1.  1. ]
ACTION: create_support_ticket
REWARD: 10.0
END
```

---

## Train PPO

```bash
python3 train_agent.py
```

Model saved:
```
support_agent.zip
```

---

## Deploy to Hugging Face

```bash
openenv push --repo-id appy-fizz26/customer-support-openenv --create-pr
```

Merge PR → Settings → Factory reboot.

---

## Submission URL

```
https://appy-fizz26-customer-support-openenv.hf.space/
```

---

## Notes

- **OpenAI calls are optional**; if no key is provided, PPO + rule fallback still works.
- The project is fully compatible with OpenEnv and HF Spaces Docker runtime.
