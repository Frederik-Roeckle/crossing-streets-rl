import numpy as np
import gymnasium as gym


class GridWorldEnv(gym.Env):
    ACTION_MOVE_UP = 0
    ACTION_MOVE_DOWN = 1
    ACTION_WAIT = 2
    ACTION_CROSS_STREET = 3
    TRAFFIC_LIGHT_TIMING = np.array([10, 3])

    def __init__(self, size=10):
        # Dimension of Grid Environment (3 x size(length))
        self.size_height = size
        self.size_width = 3

        # Init Agent and Target Location
        self.agent_position = np.array([-1, -1])
        self.target_position = np.array([-1, -1])

        self.traffic_light_1_position = np.array([-1, -1])
        self.traffic_light_2_position = np.array([-1, -1])

        # Traffic Lights Timing [0]-steps in red, [1]-steps in green
        self.traffic_light_1_timing = np.array([-1, -1])
        self.traffic_light_2_timing = np.array([-1, -1])
        
        # Current Light 0: Red, 1: Green
        self.traffic_light_1_current_light = -1
        self.traffic_light_2_current_light = -1
        

        # Define what agent can observe
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(
                low=np.array([0, 0]),
                high=np.array([self.size_width -1, self.size_height -1]),
                shape=(2,),
                dtype=int
            ),
            "target": gym.spaces.Box(
                low=np.array([0, 0]),
                high=np.array([self.size_width -1, self.size_height -1]),
                shape=(2,),
                dtype=int
            ),
            "traffic_light_1_position": gym.spaces.Box(
                low = np.array([1, 0]),
                high = np.array([1, self.size_height -1]),
                shape = (2, ),
                dtype = int
            ),
            "traffic_light_2_position": gym.spaces.Box(
                low = np.array([1, 0]),
                high = np.array([1, self.size_height -1]),
                shape = (2, ),
                dtype = int
            ),
            "traffic_light_1_current_light": gym.spaces.Discrete(
                n=2,
            ),
            "traffic_light_2_current_light": gym.spaces.Discrete(
                n=2,
            )
        })

        # Define what actions are available (Up, Down, Wait, Cross)
        self._action_to_direction = {
                self.ACTION_MOVE_UP: np.array([0, 1]),     # Up
                self.ACTION_MOVE_DOWN: np.array([0, -1]),    # Down
                self.ACTION_WAIT: np.array([0, 0]),     # Wait
                self.ACTION_CROSS_STREET: np.array([2, 0])      # Cross Road 
        }

        self.action_space = gym.spaces.Discrete(4)

    def _get_obs(self):
        """Convert internal state to observational form"""
        return {"agent": self.agent_position, 
                "target": self.target_position, 
                "traffic_light_1_position": self.traffic_light_1_position,
                "traffic_light_2_position": self.traffic_light_2_position,
                "traffic_light_1_current_light": self.traffic_light_1_current_light,
                "traffic_light_2_current_light": self.traffic_light_2_current_light,
            }

    def _get_info(self):
        """Provide auxillary information for the debugger"""
        return {
            "distance": np.linalg.norm(self.agent_position - self.target_position, ord=1)
        }
    

    def reset(self, seed=None, options=None):
        """Start a new episode"""
        super().reset(seed=seed)

        # Place Agent on the start
        self.agent_position = np.array([0, 0])

        # Place Target
        self.target_position = np.array([2, 9])

        # Place Traffic Lights
        self.traffic_light_1_position = np.array([1, 3])
        self.traffic_light_2_position = np.array([1, 6])

        # Set Traffic Light Cycles
        self.traffic_light_1_timing, self.traffic_light_1_current_light = self.traffic_light_step(self.TRAFFIC_LIGHT_TIMING.copy(), 0)
        self.traffic_light_2_timing, self.traffic_light_2_current_light = self.traffic_light_step(self.TRAFFIC_LIGHT_TIMING.copy(), 0)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    

    def traffic_light_step(self, timing, current_light):
        """One step for the traffic light"""
        if np.array_equal(timing, np.array([0, 0])):
            return self.TRAFFIC_LIGHT_TIMING.copy(), 0
        if timing[current_light] == 0:
            if current_light == 0:
                current_light = 1
                timing[current_light] -= 1
                return timing, current_light
        timing[current_light] -= 1
        return timing, current_light


    def step(self, action):
        reward = 0
        # Translate action to direction
        direction = self._action_to_direction[action]

        # if agent selects ACTION_CROSS_STREET and is not allowed to cross the street there, then he waits
        if (action == self.ACTION_CROSS_STREET):
            if((self.agent_position[1] == self.traffic_light_1_position[1] and self.traffic_light_1_current_light == 1) or
                (self.agent_position[1] == self.traffic_light_2_position[1] and self.traffic_light_2_current_light == 1)):
                pass
            else:
                direction = self._action_to_direction[self.ACTION_WAIT]  
                reward -= 10
        
        # Move agent to new position and clip if moves outside of space
        self.agent_position = np.clip(self.agent_position + direction, 0, np.array([self.size_width - 1, self.size_height -1]))

        # One step for the traffic light
        self.traffic_light_1_timing, self.traffic_light_1_current_light = self.traffic_light_step(self.traffic_light_1_timing, self.traffic_light_1_current_light)
        self.traffic_light_2_timing, self.traffic_light_2_current_light = self.traffic_light_step(self.traffic_light_2_timing, self.traffic_light_2_current_light)

        # Check if episode should be terminated as agent reached target
        terminated = np.array_equal(self.agent_position, self.target_position)

        # Truncation of the environment, e.g. step limit or time limit after which automatically end
        truncated = False

        # Set the reward
        reward += 1 if terminated else 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
