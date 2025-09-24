import gymnasium as gym
from collections import defaultdict
from Agents.Agent import Agent
import numpy as np

class SARSAAgent(Agent):
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        super().__init__()
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n, dtype=np.float64))  # ✅ Explicit dtype
        self.lr = float(learning_rate)  # ✅ Ensure scalar
        self.discount_factor = float(discount_factor)  # ✅ Ensure scalar

        self.epsilon = float(initial_epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.final_epsilon = float(final_epsilon)

        self.training_error = []
    
    def get_action(self, obs):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            obs_key = self.obs_to_key(obs)
            return int(np.argmax(self.q_values[obs_key]))
    
    def update(self, obs, action, reward, terminated, next_obs, next_action):
        """SARSA update - uses actual next action taken"""
        # ✅ Safety checks
        if action is None:
            print("Warning: action is None, skipping update")
            return
            
        obs_key = self.obs_to_key(obs)
        action = int(action)
        reward = float(reward)  # ✅ Ensure scalar
        
        # ✅ Get current Q-value as scalar
        current_q_value = float(self.q_values[obs_key][action])
        
        if not terminated and next_action is not None:
            next_action = int(next_action)
            next_obs_key = self.obs_to_key(next_obs)
            
            # ✅ Get next Q-value as scalar
            next_q_value = float(self.q_values[next_obs_key][next_action])
            target = reward + self.discount_factor * next_q_value
        else:
            target = reward
        
        # ✅ Calculate temporal difference as scalar
        temporal_difference = float(target - current_q_value)
        
        # ✅ Calculate new Q-value and assign directly
        new_q_value = current_q_value + self.lr * temporal_difference
        self.q_values[obs_key][action] = new_q_value
        
        # ✅ Append scalar to training error
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def obs_to_key(self, obs):
        """Convert observation dict to a hashable tuple key"""
        return (
            tuple(int(x) for x in obs["agent"]),  # ✅ Ensure integers
            tuple(int(x) for x in obs["target"]),
            tuple(int(x) for x in obs["traffic_light_1_position"]),
            tuple(int(x) for x in obs["traffic_light_2_position"]),
            tuple(int(x) for x in obs["traffic_light_3_position"]),
            int(obs["traffic_light_1_current_light"]),
            int(obs["traffic_light_2_current_light"]),
            int(obs["traffic_light_3_current_light"])
        )