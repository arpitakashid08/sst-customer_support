# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gradio demo for the Customer Support AI Environment."""

from __future__ import annotations

from typing import Dict, List, Tuple

import gradio as gr
import plotly.graph_objects as go

from my_env.agent import PPOAgent
from my_env.support_env import (
    infer_scenario_from_message,
    reward_for_action,
    encode_observation,
)

SCENARIOS: List[Dict[str, str]] = [
    {
        "title": "Refund Request",
        "goal": "Customer requires a refund for a recent purchase.",
        "message": "I want a refund for the last purchase.",
        "badge": "Refund",
    },
    {
        "title": "Failed Payment",
        "goal": "Resolve duplicated or failed payment issue.",
        "message": "My payment failed but the money was deducted.",
        "badge": "Payment",
    },
    {
        "title": "Angry Customer",
        "goal": "Escalate frustrated user to a human agent.",
        "message": "This is unacceptable. I want to complain now.",
        "badge": "Complaint",
    },
]

AGENT = PPOAgent()


def _build_reward_plot(rewards: List[int]) -> go.Figure:
    steps = list(range(1, len(rewards) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=rewards,
            mode="lines+markers",
            line=dict(color="#6C8CFF", width=3),
            marker=dict(size=8),
        )
    )
    fig.update_layout(
        title="Reward Progression",
        xaxis_title="Step",
        yaxis_title="Reward",
        height=250,
        margin=dict(l=30, r=10, t=50, b=30),
        template="plotly_white",
        paper_bgcolor="#111318",
        plot_bgcolor="#111318",
        font=dict(color="#E6E6E6"),
        xaxis=dict(gridcolor="#23252B"),
        yaxis=dict(gridcolor="#23252B"),
    )
    return fig


def _format_chat(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for item in history:
        messages.append({"role": "user", "content": item["user"]})
        messages.append({"role": "assistant", "content": item["agent"]})
    return messages


def _generate_response(action_label: str) -> str:
    if action_label == "reply_to_customer":
        return "Thanks for reaching out. Here's the information you requested."
    if action_label == "create_support_ticket":
        return "I've created a support ticket and our team will follow up shortly."
    return "I'm escalating this to a human support specialist for faster resolution."


def _run_agent(
    message: str,
    history: List[Dict[str, str]],
    rewards: List[int],
    step_count: int,
    scenario_label: str,
) -> Tuple[
    List[Dict[str, str]],
    str,
    str,
    str,
    str,
    float,
    Dict[str, float],
    List[List[str]],
    go.Figure,
    List[Dict[str, str]],
    List[int],
    int,
    str,
]:
    if not message.strip():
        return (
            _format_chat(history),
            "",
            "general",
            "medium",
            "reply_to_customer",
            0.0,
            {},
            [[h["user"], h["issue"], h["priority"], h["action"], h["reward"]] for h in history],
            _build_reward_plot(rewards),
            history,
            rewards,
            step_count,
            scenario_label,
        )

    if history is None:
        history = []
    if rewards is None:
        rewards = []

    if step_count is None:
        step_count = 0

    try:
        scenario = infer_scenario_from_message(message)
        issue_type = scenario.issue_type
        priority = scenario.priority
        scenario_label = scenario.name

        agent_result = AGENT.predict(encode_observation(scenario), deterministic=False)
        action = agent_result.action_label
        response = _generate_response(action)
        reward = reward_for_action(scenario, agent_result.action_id)
        action_probs = agent_result.action_probs
    except Exception as exc:  # prevent Gradio error panels
        issue_type = "general"
        priority = "medium"
        scenario_label = "Unknown"
        action = "reply_to_customer"
        response = f"Agent error: {exc}"
        reward = 0.0
        action_probs = {}

    step_count += 1
    rewards.append(int(reward))
    history.append(
        {
            "user": message,
            "agent": response,
            "issue": issue_type,
            "priority": priority,
            "action": action,
            "reward": str(int(reward)),
        }
    )

    table = [[h["user"], h["issue"], h["priority"], h["action"], h["reward"]] for h in history]

    return (
        _format_chat(history),
        response,
        issue_type,
        priority,
        action,
        reward,
        action_probs,
        table,
        _build_reward_plot(rewards),
        history,
        rewards,
        step_count,
        scenario_label,
    )


def _select_scenario(index: int) -> Tuple[str, str]:
    scenario = SCENARIOS[index]
    return scenario["message"], scenario["title"]


def _reset_state() -> Tuple[
    List[Dict[str, str]],
    str,
    str,
    str,
    str,
    float,
    Dict[str, float],
    List[List[str]],
    go.Figure,
    List[Dict[str, str]],
    List[int],
    int,
    str,
]:
    empty_history: List[Dict[str, str]] = []
    empty_rewards: List[int] = []
    return (
        [],
        "",
        "general",
        "medium",
        "reply_to_customer",
        0.0,
        {},
        [],
        _build_reward_plot(empty_rewards),
        empty_history,
        empty_rewards,
        0,
        "",
    )


def _metric_bar(label: str, value: int, color: str) -> str:
    return f"""
    <div class="metric">
      <label>{label} <span class="metric-score">{value}%</span></label>
      <div class="metric-bar">
        <span style="width:{value}%; background:{color};"></span>
      </div>
    </div>
    """


def build_demo() -> gr.Blocks:
    css = """
    body { background: #0B0D12; }
    .app-shell { background: linear-gradient(135deg, #0B0D12 0%, #11151D 100%); }
    .panel { background: #111318; border: 1px solid #23252B; border-radius: 12px; padding: 16px; }
    .panel-title { font-size: 14px; color: #8F96A3; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.08em; }
    .scenario { background: #151922; border: 1px solid #262B36; border-radius: 12px; padding: 14px; margin-bottom: 12px; }
    .scenario h4 { margin: 0 0 6px 0; color: #F2F4F8; }
    .scenario p { margin: 0 0 10px 0; color: #A7AEBB; font-size: 13px; }
    .badge { display: inline-block; background: #223055; color: #9AB1FF; padding: 2px 8px; border-radius: 999px; font-size: 12px; }
    .header { text-align: center; margin-bottom: 16px; }
    .header h1 { margin: 0; color: #F2F4F8; font-size: 26px; }
    .header p { margin: 6px 0 0; color: #9AA3B2; }
    .metric { margin-bottom: 14px; }
    .metric label { display:block; color:#9AA3B2; font-size:12px; margin-bottom:6px; }
    .metric-score { float: right; color: #D0D4DC; font-weight: 600; }
    .metric .value { color:#F2F4F8; font-size:20px; font-weight:600; }
    .metric-bar { height: 8px; background: #1E2330; border-radius: 999px; overflow: hidden; }
    .metric-bar span { display:block; height:100%; background: linear-gradient(90deg, #6C8CFF, #5AE0C7); }
    """

    with gr.Blocks(css=css, title="Customer Support AI Environment") as demo:
        with gr.Column(elem_classes=["app-shell"]):
            with gr.Column(elem_classes=["header"]):
                gr.Markdown("# Customer Support AI Environment")
                gr.Markdown(
                    "Simulate customer support interactions, track rewards, and evaluate agent decisions."
                )

            history_state = gr.State([])
            reward_state = gr.State([])
            step_state = gr.State(0)
            scenario_state = gr.State("")

            scenario_buttons: List[gr.Button] = []

            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("**1. Environment Scenarios**")
                    with gr.Column(elem_classes=["panel"]):
                        for idx, scenario in enumerate(SCENARIOS):
                            with gr.Column(elem_classes=["scenario"]):
                                gr.Markdown(f"**{scenario['title']}**")
                                gr.Markdown(scenario["goal"])
                                gr.HTML(f"<span class='badge'>{scenario['badge']}</span>")
                                scenario_buttons.append(gr.Button(
                                    "Deploy Scenario",
                                    variant="secondary",
                                ))

                with gr.Column(scale=5):
                    gr.Markdown("**2. AI Agent Interaction**")
                    with gr.Column(elem_classes=["panel"]):
                        with gr.Row():
                            episode = gr.Textbox(
                                label="Episode",
                                value="1",
                                interactive=False,
                                scale=1,
                            )
                            step = gr.Textbox(
                                label="Step",
                                value="0",
                                interactive=False,
                                scale=1,
                            )
                            scenario_label = gr.Textbox(
                                label="Scenario",
                                value="",
                                interactive=False,
                                scale=2,
                            )

                        chat = gr.Chatbot(height=320, label="Agent Interaction")
                        customer_query = gr.Textbox(
                            label="Customer Query",
                            placeholder="Type a customer issue or deploy a scenario...",
                        )
                        with gr.Row():
                            send_btn = gr.Button("Send", variant="primary")
                            clear_btn = gr.Button("Clear")

                        agent_response = gr.Textbox(
                            label="Agent Response",
                            interactive=False,
                            lines=3,
                        )
                        detected_issue = gr.Textbox(
                            label="Detected Issue Type",
                            interactive=False,
                        )
                        detected_priority = gr.Textbox(
                            label="Priority",
                            interactive=False,
                        )
                        action_taken = gr.Textbox(
                            label="Action Taken",
                            interactive=False,
                        )
                        reward_earned = gr.Number(
                            label="Reward Earned",
                            precision=0,
                            interactive=False,
                        )
                        action_probs = gr.JSON(
                            label="PPO Action Probabilities",
                            value={},
                        )

                with gr.Column(scale=3):
                    gr.Markdown("**3. Evaluation Metrics**")
                    with gr.Column(elem_classes=["panel"]):
                        total_reward = gr.Number(
                            label="OpenEnv Reward Breakdown",
                            value=0,
                            precision=2,
                            interactive=False,
                        )
                        gr.HTML(_metric_bar("Politeness", 100, "#5AE0C7"))
                        gr.HTML(_metric_bar("Resolution Quality", 100, "#6C8CFF"))
                        gr.HTML(_metric_bar("Escalation Risk", 30, "#FF6B6B"))
                        gr.HTML(_metric_bar("Response Clarity", 85, "#8BE28B"))

                        reward_plot = gr.Plot(label="Reward Progression")

            with gr.Row():
                with gr.Column(elem_classes=["panel"]):
                    history_table = gr.Dataframe(
                        headers=[
                            "Customer Query",
                            "Issue Type",
                            "Priority",
                            "Action Taken",
                            "Reward",
                        ],
                        label="Interaction History",
                        interactive=False,
                    )

            def _sync_deploy(message: str, title: str):
                return message, title

            def _update_step(step_count: int) -> str:
                return str(step_count)

            def _update_total_reward(rewards: List[int]) -> float:
                return float(sum(rewards))

            send_btn.click(
                _run_agent,
                inputs=[customer_query, history_state, reward_state, step_state, scenario_state],
                outputs=[
                    chat,
                    agent_response,
                    detected_issue,
                    detected_priority,
                    action_taken,
                    reward_earned,
                    action_probs,
                    history_table,
                    reward_plot,
                    history_state,
                    reward_state,
                    step_state,
                    scenario_state,
                ],
            ).then(
                _update_step,
                inputs=[step_state],
                outputs=[step],
            ).then(
                _update_total_reward,
                inputs=[reward_state],
                outputs=[total_reward],
            )

            clear_btn.click(
                _reset_state,
                outputs=[
                    chat,
                    agent_response,
                    detected_issue,
                    detected_priority,
                    action_taken,
                    reward_earned,
                    action_probs,
                    history_table,
                    reward_plot,
                    history_state,
                    reward_state,
                    step_state,
                    scenario_state,
                ],
            ).then(
                _update_step,
                inputs=[step_state],
                outputs=[step],
            ).then(
                _update_total_reward,
                inputs=[reward_state],
                outputs=[total_reward],
            )

            # Scenario buttons wiring
            for idx, button in enumerate(scenario_buttons):
                button.click(
                    lambda idx=idx: _select_scenario(idx),
                    inputs=None,
                    outputs=[customer_query, scenario_label],
                    api_name=None,
                    scroll_to_output=False,
                )

    return demo


if __name__ == "__main__":
    build_demo().launch()
