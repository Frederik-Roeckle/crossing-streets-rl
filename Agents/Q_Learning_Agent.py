from collections import defaultdict
import gymnasium as gym
import numpy as np
from Agents.Agent import Agent


class QLearningAgent(Agent):
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
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
    

    def get_action(self, obs):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            obs_key = self.obs_to_key(obs)
            return int(np.argmax(self.q_values[obs_key]))
        
    
    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        next_obs_key = self.obs_to_key(next_obs)
        obs_key = self.obs_to_key(obs)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs_key])

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[obs_key][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[obs_key][action] = (
            self.q_values[obs_key][action] + self.lr * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


    def obs_to_key(self, obs):
        """Convert observation dict to a hashable tuple key"""
        return (
            tuple(obs["agent"]),
            tuple(obs["target"]),
            tuple(obs["traffic_light_1_position"]),
            tuple(obs["traffic_light_2_position"]),
            obs["traffic_light_1_current_light"],
            obs["traffic_light_2_current_light"]
        )
    