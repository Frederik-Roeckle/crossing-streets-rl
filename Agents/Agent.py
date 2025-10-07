import pickle
from collections import defaultdict
import numpy as np
import json

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
            # 'training_error': self.training_error
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
        # self.training_error = agent_state['training_error']
        print(f"Agent state loaded from {filepath}")

    def save_q_table_json(self, filepath):
        """Save the Q-table to a JSON file"""
        try:
            # Convert Q-table to JSON-serializable format
            q_table_json = {}
            
            for state, values in self.q_values.items():
                # Convert state tuple to string key
                state_key = str(state)
                
                # Convert numpy array to list
                if isinstance(values, np.ndarray):
                    q_table_json[state_key] = values.tolist()
                else:
                    q_table_json[state_key] = list(values)
            
            # Save as JSON
            with open(filepath, 'w') as f:
                json.dump(q_table_json, f, indent=2)
            
            print(f"Q-table saved to JSON: {filepath}")
            print(f"States saved: {len(q_table_json)}")
            
        except Exception as e:
            print(f"Error saving Q-table to JSON: {e}")
            raise

    def load_q_table_json(self, filepath):
        """Load Q-table from a JSON file"""
        try:
            with open(filepath, 'r') as f:
                q_table_json = json.load(f)
            
            # Convert back to defaultdict with numpy arrays
            self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))
            
            for state_key, values in q_table_json.items():
                # Convert string key back to tuple
                state = eval(state_key)  # Be careful with eval in production!
                
                # Convert list back to numpy array
                self.q_values[state] = np.array(values)
            
            print(f"Q-table loaded from JSON: {filepath}")
            print(f"States loaded: {len(q_table_json)}")
            
        except Exception as e:
            print(f"Error loading Q-table from JSON: {e}")
            raise

    def save_agent_state_json(self, filepath):
        """Save complete agent state to JSON format"""
        try:
            # Convert Q-values to JSON format
            q_values_json = {}
            for state, values in self.q_values.items():
                state_key = str(state)
                if isinstance(values, np.ndarray):
                    q_values_json[state_key] = values.tolist()
                else:
                    q_values_json[state_key] = list(values)
            
            # Create agent state dictionary
            agent_state = {
                'q_values': q_values_json,
                'epsilon': float(self.epsilon),
                'lr': float(self.lr),
                'discount_factor': float(self.discount_factor),
                'final_epsilon': float(self.final_epsilon),
                # 'training_error': list(self.training_error) if hasattr(self.training_error, '__iter__') else []
            }
            
            # Save as JSON
            with open(filepath, 'w') as f:
                json.dump(agent_state, f, indent=2)
            
            print(f"Agent state saved to JSON: {filepath}")
            print(f"Q-values: {len(q_values_json)} states")
            # print(f"Training errors: {len(agent_state['training_error'])} entries")
            
        except Exception as e:
            print(f"Error saving agent state to JSON: {e}")
            raise

    def load_agent_state_json(self, filepath):
        """Load complete agent state from JSON format"""
        try:
            with open(filepath, 'r') as f:
                agent_state = json.load(f)
            
            # Restore Q-table
            self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))
            
            for state_key, values in agent_state['q_values'].items():
                # Convert string key back to tuple
                state = eval(state_key)  # Use ast.literal_eval for safer parsing
                self.q_values[state] = np.array(values)
            
            # Restore other parameters
            self.epsilon = agent_state['epsilon']
            self.lr = agent_state['lr']
            self.discount_factor = agent_state['discount_factor']
            self.final_epsilon = agent_state['final_epsilon']
            # self.training_error = agent_state['training_error']
            
            print(f"Agent state loaded from JSON: {filepath}")
            
        except Exception as e:
            print(f"Error loading agent state from JSON: {e}")
            raise