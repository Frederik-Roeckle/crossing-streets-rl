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
    max_episode_steps=500,  # Prevent infinite episodes
)

# env = gym.make("gymnasium_env/GridWorld-v0", render_mode="human")

# Training hyperparameters
learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
n_episodes = 1000         # Number of hands to practice
start_epsilon = 1.0         # Start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.1         # Always keep some exploration


# agent = SARSAAgent(
#     env=env,
#     learning_rate=learning_rate,
#     initial_epsilon=start_epsilon,
#     epsilon_decay=epsilon_decay,
#     final_epsilon=final_epsilon,
# )

def progressive_training():
    sizes = [13, 26, 52, 132]  # Progressive sizes
    agent = None
    all_stats = {'returns': [], 'lengths': [], 'training_errors': []}
    
    for i, size in enumerate(sizes):
        print(f"Training on size: {size}")
        episodes = 20000 if i == 0 else 10000
        env = gym.make("gymnasium_env/GridWorld-v0", size=size)
        env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
        
        if agent is None:
            # First training - create new agent
            agent = SARSAAgent(env,
                               learning_rate=0.1, 
                               initial_epsilon=1.0,
                               epsilon_decay=1.0 / (episodes / 2),
                               final_epsilon=0.1)
        else:
            # Transfer learning - update environment and reduce learning parameters
            agent.env = env
            agent.lr *= 0.7  # Reduce learning rate
            agent.epsilon = max(0.1, agent.epsilon * 0.5)  # Reduce exploration
            agent.epsilon_decay = agent.epsilon / (episodes / 2)
        
        # Train on current size
        train_agent(agent, env, episodes)
        
        all_stats['returns'].extend(list(env.return_queue))
        all_stats['lengths'].extend(list(env.length_queue))
        all_stats['training_errors'].extend(agent.training_error)
        
        
        # Save checkpoint
        agent.save_agent_state(f"agent_size_{size}.pkl")
        print(f"Completed size {size}: Avg reward = {np.mean(list(env.return_queue)[-100:]):.2f}")
        
        
        env.close()
    
    return agent, all_stats


def train_agent(agent, env, n_episodes):
    for _ in tqdm(range(n_episodes)):
        observation, info = env.reset()
        action = agent.get_action(observation) # chose first action
        episode_over = False

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



agent, all_stats = progressive_training()

def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    arr = np.array(arr).flatten()
    
    # ✅ Safety checks
    if len(arr) == 0:
        return np.array([])
    
    if window > len(arr):
        window = len(arr)
    
    if window <= 0:
        return arr
    
    return np.convolve(arr, np.ones(window), mode=convolution_mode) / window

# Plot results using collected statistics
if len(all_stats['returns']) > 10:
    rolling_length = min(500, len(all_stats['returns']) // 4)
    
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # Episode rewards
    axs[0].set_title("Episode rewards")
    if len(all_stats['returns']) >= rolling_length:
        reward_moving_average = get_moving_avgs(all_stats['returns'], rolling_length, "valid")
        if len(reward_moving_average) > 0:
            axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # Episode lengths
    axs[1].set_title("Episode lengths")
    if len(all_stats['lengths']) >= rolling_length:
        length_moving_average = get_moving_avgs(all_stats['lengths'], rolling_length, "valid")
        if len(length_moving_average) > 0:
            axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # Training error
    axs[2].set_title("Training Error")
    if len(all_stats['training_errors']) >= rolling_length:
        training_error_moving_average = get_moving_avgs(all_stats['training_errors'], rolling_length, "same")
        if len(training_error_moving_average) > 0:
            axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    plt.show()
else:
    print("Not enough data for plotting")

# Save final agent
agent.save_agent_state("sarsa_agent_final.pkl")
agent.save_q_table("sarsa_q_table_final.pkl")