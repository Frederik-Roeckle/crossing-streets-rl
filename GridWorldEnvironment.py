from enum import Enum
import numpy as np
import gymnasium as gym
import pygame


class Actions(Enum):
    ACTION_MOVE_UP = 0
    ACTION_MOVE_DOWN = 1
    ACTION_WAIT = 2
    ACTION_CROSS_STREET = 3

class GridWorldEnv(gym.Env):
    
    TRAFFIC_LIGHT_TIMING = np.array([10, 3])

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=10):
        # Dimension of Grid Environment (3 x size(length))
        self.size_height = size
        self.size_width = 3

        # for rendering
        self.window_size_height = 512
        self.window_size_width = 300

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
                Actions.ACTION_MOVE_UP.value: np.array([0, 1]),     # Up
                Actions.ACTION_MOVE_DOWN.value: np.array([0, -1]),    # Down
                Actions.ACTION_WAIT.value: np.array([0, 0]),     # Wait
                Actions.ACTION_CROSS_STREET.value: np.array([2, 0])      # Cross Road 
        }

        self.action_space = gym.spaces.Discrete(4)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

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
        if (action == Actions.ACTION_CROSS_STREET.value):
            if((self.agent_position[1] == self.traffic_light_1_position[1] and self.traffic_light_1_current_light == 1) or
                (self.agent_position[1] == self.traffic_light_2_position[1] and self.traffic_light_2_current_light == 1)):
                pass
            else:
                direction = self._action_to_direction[Actions.ACTION_WAIT.value]  
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


    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
        
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size_width, self.window_size_height)
            )
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size_width, self.window_size_height))
        canvas.fill((255, 255, 255))
        pix_size = (
            self.window_size_width / self.size_width,
            self.window_size_height / self.size_height
        )


        # draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_size[0] * self.target_position[0],
                pix_size[1] * self.target_position[1],
                pix_size[0],
                pix_size[1]               
            )
        )

        # draw the agent
        pygame.draw.rect(
            canvas,
            (0, 0, 255),
            pygame.Rect(
                pix_size[0] * self.agent_position[0],
                pix_size[1] * self.agent_position[1],
                pix_size[0],
                pix_size[1]               
            )
        )

        # draw the traffic lights
        

        pygame.draw.rect(
            canvas,
            (255, 0, 0) if self.traffic_light_1_current_light == 0 else (0, 255, 0),
            pygame.Rect(
                pix_size[0] * self.traffic_light_1_position[0],
                pix_size[1] * self.traffic_light_1_position[1],
                pix_size[0],
                pix_size[1]               
            )
        )

        pygame.draw.rect(
            canvas,
            (255, 0, 0) if self.traffic_light_2_current_light == 0 else (0, 255, 0),
            pygame.Rect(
                pix_size[0] * self.traffic_light_2_position[0],
                pix_size[1] * self.traffic_light_2_position[1],
                pix_size[0],
                pix_size[1]               
            )
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()