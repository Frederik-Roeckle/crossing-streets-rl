# Run `pip install "gymnasium[classic-control]"` for this example.
import gymnasium as gym
from gymnasium.utils.env_checker import check_env


# Register the environment so we can create it with gym.make()
gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point="GridWorldEnvironment:GridWorldEnv",
    max_episode_steps=300,  # Prevent infinite episodes
)

env = gym.make("gymnasium_env/GridWorld-v0")

# This will catch many common issues
# try:
#     check_env(env)
#     print("Environment passes all checks!")
# except Exception as e:
#     print(f"Environment has issues: {e}")

# Reset environment to start a new episode
observation, info = env.reset(seed=111)

print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0

while not episode_over:
    # Choose an action: 0 = push cart left, 1 = push cart right
    action = env.action_space.sample()  # Random action for now - real agents will be smarter!
    # Take the action and see what happens
    observation, reward, terminated, truncated, info = env.step(action)

    print(info)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()