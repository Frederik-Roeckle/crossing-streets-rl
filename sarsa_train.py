# Run `pip install "gymnasium[classic-control]"` for this example.
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from gymnasium.utils.env_checker import check_env
from Agents.SARSA_Agent import SARSAAgent
from tqdm import tqdm


# Register the environment so we can create it with gym.make()
gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point="GridWorldEnvironment:GridWorldEnv",
    max_episode_steps=150,  # Prevent infinite episodes
)

env = gym.make("gymnasium_env/GridWorld-v0", render_mode="human")

# Training hyperparameters
learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
n_episodes = 200000         # Number of hands to practice
start_epsilon = 1.0         # Start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.1         # Always keep some exploration

env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)


agent = SARSAAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

for episode in tqdm(range(n_episodes)):
    observation, info = env.reset()
    action = agent.get_action(observation) # chose first action
    episode_over = False
    total_reward = 0

    while not episode_over:
        # Take the chosen action
        next_observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated
        
        if not episode_over:
            next_action = agent.get_action(next_observation)  # Choose next action
        else:
            next_action = None  # No next action if episode ended
        
        # SARSA update
        agent.update(observation, action, reward, terminated, next_observation, next_action)
        
        observation = next_observation
        action = next_action  # Use the action we already chose!
    
    agent.decay_epsilon()

def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

# Smooth over a 500-episode window
rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

# Episode rewards (win/loss performance)
axs[0].set_title("Episode rewards")
reward_moving_average = get_moving_avgs(
    env.return_queue,
    rolling_length,
    "valid"
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[0].set_ylabel("Average Reward")
axs[0].set_xlabel("Episode")

# Episode lengths (how many actions per hand)
axs[1].set_title("Episode lengths")
length_moving_average = get_moving_avgs(
    env.length_queue,
    rolling_length,
    "valid"
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[1].set_ylabel("Average Episode Length")
axs[1].set_xlabel("Episode")

# Training error (how much we're still learning)
axs[2].set_title("Training Error")
training_error_moving_average = get_moving_avgs(
    agent.training_error,
    rolling_length,
    "same"
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
axs[2].set_ylabel("Temporal Difference Error")
axs[2].set_xlabel("Step")

plt.tight_layout()
plt.show()

agent.save_agent_state("sarsa_agent.pkl")
agent.save_q_table("q_learning_table.pkl")

# print(f"Episode finished! Total reward: {total_reward}")
env.close()