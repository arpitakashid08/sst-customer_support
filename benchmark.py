# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Simple benchmark script for the customer support environment."""

from my_env.agent import RuleBasedAgent
from my_env.server.my_env_environment import MyEnvironment


def run_episode() -> None:
    env = MyEnvironment()
    agent = RuleBasedAgent()

    observation = env.reset()
    action = agent.act(observation)
    observation = env.step(action)

    print("Customer message:", observation.customer_message)
    print("Issue type:", observation.issue_type)
    print("Priority:", observation.priority)
    print("Action taken:", action.action)
    print("Reward:", observation.reward)


if __name__ == "__main__":
    run_episode()
