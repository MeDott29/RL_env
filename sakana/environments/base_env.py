import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import time
from abc import ABC, abstractmethod

class BaseEnvironment(gym.Env, ABC):
    """
    Base class for all environments in Sakana.
    
    This abstract class defines the interface that all environments must implement.
    It handles the basic setup for rendering and provides abstract methods that
    specific environments need to implement.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, width=400, height=400, render_mode=None, max_steps=1000):
        """
        Initialize the base environment.
        
        Args:
            width (int): Width of the environment in pixels
            height (int): Height of the environment in pixels
            render_mode (str): Mode for rendering ('human', 'rgb_array', or None)
            max_steps (int): Maximum number of steps per episode
        """
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.steps = 0
        
        # These will be defined in the child classes
        self.observation_space = None
        self.action_space = None
        
        # Initialize pygame for rendering if needed
        self.window = None
        self.clock = None
        if self.render_mode is not None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((width, height))
            self.clock = pygame.time.Clock()
    
    @abstractmethod
    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        
        Args:
            seed (int, optional): Random seed for reproducibility
            options (dict, optional): Additional options for resetting
            
        Returns:
            observation: The initial observation
            info: Additional information
        """
        pass
    
    @abstractmethod
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            observation: The new observation
            reward: The reward for the action
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated (e.g., due to max steps)
            info: Additional information
        """
        pass
    
    @abstractmethod
    def _get_observation(self):
        """
        Get the current observation of the environment.
        
        Returns:
            observation: The current observation
        """
        pass
    
    @abstractmethod
    def _get_info(self):
        """
        Get additional information about the environment.
        
        Returns:
            info: A dictionary of additional information
        """
        pass
    
    def render(self):
        """
        Render the environment.
        
        Returns:
            If render_mode is 'rgb_array', returns an RGB array of the scene.
            If render_mode is 'human', renders to the pygame window and returns None.
        """
        if self.render_mode is None:
            return
        
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width, self.height))
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.width, self.height))
        canvas.fill((255, 255, 255))
        
        # This method should be implemented by child classes to draw environment-specific elements
        self._render_environment(canvas)
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    @abstractmethod
    def _render_environment(self, canvas):
        """
        Render environment-specific elements to the canvas.
        
        Args:
            canvas: Pygame surface to draw on
        """
        pass
    
    def close(self):
        """
        Close the environment and clean up resources.
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
    
    def render_episode(self, agent, max_steps=1000, delay=0.05):
        """
        Render a full episode with the given agent.
        
        Args:
            agent: The agent to use for actions
            max_steps (int): Maximum number of steps to render
            delay (float): Delay between frames in seconds
        """
        obs, info = self.reset()
        done = False
        truncated = False
        total_reward = 0
        
        for _ in range(max_steps):
            if done or truncated:
                break
                
            action = agent.select_action(obs)
            obs, reward, done, truncated, info = self.step(action)
            total_reward += reward
            
            self.render()
            time.sleep(delay)
        
        print(f"Episode finished with total reward: {total_reward}")
        self.close()
        return total_reward 