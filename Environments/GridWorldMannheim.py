from enum import Enum
import numpy as np
import gymnasium as gym
import pygame


class Actions(Enum):
    ACTION_MOVE_LEFT = 0
    ACTION_MOVE_RIGHT = 1
    ACTION_MOVE_UP = 2
    ACTION_MOVE_DOWN = 3
    ACTION_WAIT = 4

class Locations(Enum):
    CASTLE = (27, 0)
    MENSA = (15, 0)
    STORE = (27, 2)
    CAFFEE = (5, 2)
    LIBRARY = (0, 2)

class Crossings:
    # Traffic Light 1
    TL_1_POSITION = (1, 1)
    TL_TIMING_1 = (12, 2)

    # TL2 has an orthogonal traffic light that works inverse to the 
    TL_2_POSITION = (15, 1)
    TL_2b_POSITION = (14, 2)
    TL_TIMING_2 = (12, 2)
    TL_TIMING_2b = (2, 12)

    # Longer Traffic Light Timing for TL3 as it has a small extra TL
    TL_3_POSITION = (26, 1)
    TL_TIMING_3 = (15, 3)
    
    # Small Street Crossing (SC) with a probability of .95 of being free to cross
    SC_3_Position = (25, 2)
    SC_3_CHANCE_2_CROSS = 0.9
    



class GWMannheimEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, size=28):
        # Dimension of Grid Environment (3 x size(length))
        self.size_height = 3
        self.size_width = size

        # for rendering
        self.window_size_height = 300
        self.window_size_width = 792

        # Init Agent and Target Location
        self.agent_position = np.array([-1, -1])
        self.target_position = np.array([-1, -1])

        self.traffic_light_1_position = np.array([-1, -1])
        self.traffic_light_2_position = np.array([-1, -1])
        self.traffic_light_2b_position = np.array([-1, -1])
        self.traffic_light_3_position = np.array([-1, -1])
        self.street_crossing_3_position = np.array([-1, -1])

        # Traffic Lights Timing [0]-steps in red, [1]-steps in green
        self.traffic_light_1_timing = np.array([-1, -1])
        self.traffic_light_2_timing = np.array([-1, -1])
        self.traffic_light_2b_timing = np.array([-1, -1])
        self.traffic_light_3_timing = np.array([-1, -1])
        
        # Current Light 0: Red, 1: Green
        self.traffic_light_1_current_light = -1
        self.traffic_light_2_current_light = -1
        self.traffic_light_2b_current_light = -1
        self.traffic_light_3_current_light = -1
        self.street_crossing_3_status = -1
        
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
            "traffic_light_2b_position": gym.spaces.Box(
                low = np.array([1, 0]),
                high = np.array([1, self.size_height -1]),
                shape = (2, ),
                dtype = int
            ),
            "traffic_light_3_position": gym.spaces.Box(
                low = np.array([1, 0]),
                high = np.array([1, self.size_height -1]),
                shape = (2, ),
                dtype = int
            ),
            "street_crossing_3_position": gym.spaces.Box(
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
            ),
            "traffic_light_2b_current_light": gym.spaces.Discrete(
                n=2,
            ),
            "traffic_light_3_current_light": gym.spaces.Discrete(
                n=2,
            ),
            # crossable Y/n
            "street_crossing_3_status": gym.spaces.Discrete(
                n=2,
            )
        })

        # Define what actions are available (Left, Right, Up, Down, Wait)
        self._action_to_direction = {
                Actions.ACTION_MOVE_LEFT.value: np.array([-1, 0]),    # Left
                Actions.ACTION_MOVE_RIGHT.value: np.array([1, 0]),    # Right
                Actions.ACTION_MOVE_UP.value: np.array([0, 1]),       # Up
                Actions.ACTION_MOVE_DOWN.value: np.array([0, -1]),    # Down
                Actions.ACTION_WAIT.value: np.array([0, 0]),          # Wait
        }

        self.action_space = gym.spaces.Discrete(5)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        """Convert internal state to observational form"""
        return {"agent": self.agent_position, 
                "target": self.target_position, 
                "traffic_light_1_position": self.traffic_light_1_position,
                "traffic_light_2_position": self.traffic_light_2_position,
                "traffic_light_2b_position": self.traffic_light_2b_position,
                "traffic_light_3_position": self.traffic_light_3_position,
                "street_crossing_3_position": self.street_crossing_3_position,
                "traffic_light_1_current_light": self.traffic_light_1_current_light,
                "traffic_light_2_current_light": self.traffic_light_2_current_light,
                "traffic_light_2b_current_light": self.traffic_light_2b_current_light,
                "traffic_light_3_current_light": self.traffic_light_3_current_light,
                "street_crossing_3_status": self.street_crossing_3_status,
            }

    def _get_info(self):
        """Provide auxillary information for the debugger"""
        return {
            "distance": np.linalg.norm(self.agent_position - self.target_position, ord=1)
        }
    

    def reset(self, seed=None, options=None):
        """Start a new episode"""
        super().reset(seed=seed)
        
        agent_pos, target_pos = self.sample_start_target_locations()
        self.agent_position = np.array(agent_pos)
        self.target_position = np.array(target_pos)

        self.traffic_light_1_position = np.array(Crossings.TL_1_POSITION)
        self.traffic_light_2_position = np.array(Crossings.TL_2_POSITION)
        self.traffic_light_2b_position = np.array(Crossings.TL_2b_POSITION)
        self.traffic_light_3_position = np.array(Crossings.TL_3_POSITION)
        self.street_crossing_3_position = np.array(Crossings.SC_3_Position)

        # Set Traffic Light Cycles
        initial_timing_light_1 = np.array([self.np_random.integers(0, Crossings.TL_TIMING_1[0], dtype=int), 
                                           self.np_random.integers(0, Crossings.TL_TIMING_1[1], dtype=int)])
        initial_timing_light_2 = np.array([self.np_random.integers(0, Crossings.TL_TIMING_2[0], dtype=int), 
                                           self.np_random.integers(0, Crossings.TL_TIMING_2[1], dtype=int)])
        initial_timing_light_2b = np.array([self.np_random.integers(0, Crossings.TL_TIMING_2b[0], dtype=int), 
                                            self.np_random.integers(0, Crossings.TL_TIMING_2b[1], dtype=int)])
        initial_timing_light_3 = np.array([self.np_random.integers(0, Crossings.TL_TIMING_3[0], dtype=int), 
                                           self.np_random.integers(0, Crossings.TL_TIMING_3[1], dtype=int)])
        
        self.traffic_light_1_timing, self.traffic_light_1_current_light = self.traffic_light_step(initial_timing_light_1, 0, tl="TL1")
        self.traffic_light_2_timing, self.traffic_light_2_current_light = self.traffic_light_step(initial_timing_light_2, 0, tl="TL2")
        self.traffic_light_2b_timing , self.traffic_light_2b_current_light = self.traffic_light_step(initial_timing_light_2b, 0, tl="TL2b")
        self.traffic_light_3_timing, self.traffic_light_3_current_light = self.traffic_light_step(initial_timing_light_3, 0, tl="TL3")

        self.street_crossing_3_status = 1 if self.np_random.random() < Crossings.SC_3_CHANCE_2_CROSS else 0
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    

    def traffic_light_step(self, timing, current_light, tl=None):
        """One step for the traffic light"""
        if np.array_equal(timing, np.array([0, 0])):
            if tl == "TL1":
                return np.array(Crossings.TL_TIMING_1), 0
            elif tl == "TL2":
                return np.array(Crossings.TL_TIMING_2), 0
            elif tl == "TL2b":
                return np.array(Crossings.TL_TIMING_2b), 0
            elif tl == "TL3":
                return np.array(Crossings.TL_TIMING_3), 0
            else:   # Fallback
                raise Exception
        if timing[current_light] == 0:
            if current_light == 0:
                current_light = 1
                timing[current_light] -= 1
                return timing, current_light
        timing[current_light] -= 1
        return timing, current_light


    def step(self, action):
        # each step longer gets a slight negative reward
        reward = -.1
        # Translate action to direction
        direction = self._action_to_direction[action]

        # Agent Move UP/DOWN
        if (action == Actions.ACTION_MOVE_UP.value or action == Actions.ACTION_MOVE_DOWN.value):    # The agent crosses the street vertically correctly
            if((self.agent_position[0] == self.traffic_light_1_position[0] and self.traffic_light_1_current_light == 1) or
                (self.agent_position[0] == self.traffic_light_2_position[0] and self.traffic_light_2_current_light == 1) or
                (self.agent_position[0] == self.traffic_light_3_position[0] and self.traffic_light_3_current_light == 1)):
                # Let agent cross the street and doesnt let him stop at an [x, 1] field
                direction = 2 * direction       
                # reward += 1
            else:   # The agent wants to illegally cross the street
                direction = self._action_to_direction[Actions.ACTION_WAIT.value]  
                reward -= 10
        
        # Agent Move LEFT/RIGHT -> check StreetCrossing and TL2b
        if ((action == Actions.ACTION_MOVE_LEFT.value or action == Actions.ACTION_MOVE_RIGHT.value) and self.agent_position[1] == 2):
            if np.array_equal(self.agent_position + direction, self.traffic_light_2b_position):   # TL2b check
                if self.traffic_light_2b_current_light == 0:    # illegal move
                    direction = self._action_to_direction[Actions.ACTION_WAIT.value]  
                    reward -= 10
            
            if np.array_equal(self.agent_position + direction, self.street_crossing_3_position):  # SC check
                if self.street_crossing_3_status == 0:          # illegal move
                    direction = self._action_to_direction[Actions.ACTION_WAIT.value]  
                    reward -= 10

        
        # Move agent to new position and clip if moves outside of space
        self.agent_new_position = np.clip(self.agent_position + direction, 0, np.array([self.size_width - 1, self.size_height -1]))
        
        # if clipping was necessary, then penalty for out of bound
        if not np.array_equal((self.agent_position + direction), self.agent_new_position):
            reward -= 1
        
        # If Agent moves closer to Target, gets reward
        # if np.linalg.norm(self.agent_new_position - self.target_position, ord=1) < np.linalg.norm(self.agent_position - self.target_position, ord=1):
        #     reward += .1
        self.agent_position = self.agent_new_position

        # One step for the traffic light
        self.traffic_light_1_timing, self.traffic_light_1_current_light = self.traffic_light_step(self.traffic_light_1_timing, self.traffic_light_1_current_light, tl="TL1")
        self.traffic_light_2_timing, self.traffic_light_2_current_light = self.traffic_light_step(self.traffic_light_2_timing, self.traffic_light_2_current_light, tl="TL2")
        self.traffic_light_2b_timing, self.traffic_light_2b_current_light = self.traffic_light_step(self.traffic_light_2b_timing, self.traffic_light_2b_current_light, tl="TL2b")
        self.traffic_light_3_timing, self.traffic_light_3_current_light = self.traffic_light_step(self.traffic_light_3_timing, self.traffic_light_3_current_light, tl="TL3")

        # set street crossing status
        self.street_crossing_3_status = 1 if self.np_random.random() < Crossings.SC_3_CHANCE_2_CROSS else 0
        
        # Check if episode should be terminated as agent reached target
        terminated = np.array_equal(self.agent_position, self.target_position)

        # Truncation of the environment, e.g. step limit or time limit after which automatically end
        truncated = False

        # Reward the agent for landing on the target
        # reward += 10 if terminated else 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info



    ### HELPER FUNCTIONS
    def sample_start_target_locations(self):
        all_locations = list(Locations)
        indices = self.np_random.choice(len(all_locations), size=2, replace=False)
        return [all_locations[i].value for i in indices]


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
        # TODO: show it as a nice motive and introduce nice images for the other 
        pygame.draw.rect(
            canvas,
            (100, 100, 100),
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

        pygame.draw.rect(
            canvas,
            (255, 0, 0) if self.traffic_light_2b_current_light == 0 else (0, 255, 0),
            pygame.Rect(
                pix_size[0] * self.traffic_light_2b_position[0],
                pix_size[1] * self.traffic_light_2b_position[1],
                pix_size[0],
                pix_size[1]               
            )
        )

        pygame.draw.rect(
            canvas,
            (255, 0, 0) if self.traffic_light_3_current_light == 0 else (0, 255, 0),
            pygame.Rect(
                pix_size[0] * self.traffic_light_3_position[0],
                pix_size[1] * self.traffic_light_3_position[1],
                pix_size[0],
                pix_size[1]               
            )
        )

        # street crossing
        pygame.draw.rect(
            canvas,
            (220, 220, 120) if self.street_crossing_3_status == 1 else (0, 0, 0),
            pygame.Rect(
                pix_size[0] * self.street_crossing_3_position[0],
                pix_size[1] * self.street_crossing_3_position[1],
                pix_size[0],
                pix_size[1]               
            )
        )

        # TODO: draw all non-accessible areas


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