import numpy as np
import gymnasium as gym
import pickle
from Agents.Q_Learning_Agent import QLearningAgent
from Agents.SARSA_Agent import SARSAAgent

gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point="GridWorldEnvironment:GridWorldEnv",
    max_episode_steps=100,  # Prevent infinite episodes
)

env = gym.make("gymnasium_env/GridWorld-v0", render_mode="human")

agent = QLearningAgent(
    env=env,
    learning_rate=-1,
    initial_epsilon=-1,
    epsilon_decay=-1,
    final_epsilon=-1,
)

agent = SARSAAgent(
    env=env,
    learning_rate=-1,
    initial_epsilon=-1,
    epsilon_decay=-1,
    final_epsilon=-1,
)
agent.load_agent_state("sarsa_agent_final.pkl")

for _ in range(5):

    observation, info = env.reset()
    episode_over = False
    total_reward = 0

    while not episode_over:
        action = agent.get_action(observation)
        # Take the action and see what happens
        next_observation, reward, terminated, truncated, info = env.step(action)

        print(info)
        env.render()

        total_reward += reward
        episode_over = terminated or truncated
        observation = next_observation

env.close()
