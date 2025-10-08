import numpy as np
import gymnasium as gym
import itertools
import pickle
import Environments.GridWorldMannheim as GridWorldMannheim
from Agents.Q_Learning_Agent import QLearningAgent

gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point="Environments.GridWorldMannheim:GWMannheimEnv",
    max_episode_steps=50,  # Prevent infinite episodes
)

env = gym.make("gymnasium_env/GridWorld-v0", render_mode="human")

agent = QLearningAgent(
    env=env,
    learning_rate=-1,
    initial_epsilon=-1,
    epsilon_decay=-1,
    final_epsilon=-1,
)
agent.load_agent_state_json("./10M_Agent/q_agent.json")

locations = (
    GridWorldMannheim.Locations.CASTLE,
    GridWorldMannheim.Locations.MENSA, 
    GridWorldMannheim.Locations.STORE,
    GridWorldMannheim.Locations.CAFFEE,
    GridWorldMannheim.Locations.LIBRARY
)
location_combinations = list(itertools.permutations(locations, 2))


for location_combination_set in location_combinations:

    observation, info = env.reset(options={"start_pos": location_combination_set[0].value,
                                           "target_pos":location_combination_set[1].value})
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
