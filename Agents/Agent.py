import pickle
from collections import defaultdict
import numpy as np

class Agent():
    def save_q_table(self, filepath):
        """Save the Q-table to a pickle file"""
        # Convert defaultdict to regular dict for saving
        q_table_dict = dict(self.q_values)
        with open(filepath, 'wb') as f:
            pickle.dump(q_table_dict, f)
        print(f"Q-table saved to {filepath}")

    def load_q_table(self, filepath):
        """Load Q-table from a pickle file"""
        with open(filepath, 'rb') as f:
            q_table_dict = pickle.load(f)
        # Convert back to defaultdict
        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.q_values.update(q_table_dict)
        print(f"Q-table loaded from {filepath}")

    def save_agent_state(self, filepath):
        """Save complete agent state including hyperparameters"""
        agent_state = {
            'q_values': dict(self.q_values),
            'epsilon': self.epsilon,
            'lr': self.lr,
            'discount_factor': self.discount_factor,
            'final_epsilon': self.final_epsilon,
            'training_error': self.training_error
        }
        with open(filepath, 'wb') as f:
            pickle.dump(agent_state, f)
        print(f"Agent state saved to {filepath}")

    def load_agent_state(self, filepath):
        """Load complete agent state"""
        with open(filepath, 'rb') as f:
            agent_state = pickle.load(f)
        
        # Restore Q-table
        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.q_values.update(agent_state['q_values'])
        
        # Restore other parameters
        self.epsilon = agent_state['epsilon']
        self.lr = agent_state['lr']
        self.discount_factor = agent_state['discount_factor']
        self.final_epsilon = agent_state['final_epsilon']
        self.training_error = agent_state['training_error']
        print(f"Agent state loaded from {filepath}")