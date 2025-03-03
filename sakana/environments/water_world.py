import numpy as np
import pygame
from gymnasium import spaces
from .base_env import BaseEnvironment
import math

class WaterWorld(BaseEnvironment):
    """
    WaterWorld environment where agents navigate in a fluid-like environment.
    
    The agent is represented as a circular entity that can apply forces to move.
    The environment contains food (positive reward) and poison (negative reward).
    The agent has sensors that detect nearby objects.
    """
    
    def __init__(self, width=400, height=400, render_mode=None, max_steps=1000,
                 num_food=5, num_poison=5, sensor_range=100, friction=0.05):
        """
        Initialize the WaterWorld environment.
        
        Args:
            width (int): Width of the environment in pixels
            height (int): Height of the environment in pixels
            render_mode (str): Mode for rendering ('human', 'rgb_array', or None)
            max_steps (int): Maximum number of steps per episode
            num_food (int): Number of food items
            num_poison (int): Number of poison items
            sensor_range (int): Range of the agent's sensors
            friction (float): Friction coefficient for agent movement
        """
        super().__init__(width, height, render_mode, max_steps)
        
        self.num_food = num_food
        self.num_poison = num_poison
        self.sensor_range = sensor_range
        self.friction = friction
        self.num_sensors = 8  # Number of sensors around the agent
        
        # Agent properties
        self.agent_radius = 15
        self.agent_pos = np.zeros(2)
        self.agent_vel = np.zeros(2)
        self.max_speed = 5.0
        self.max_force = 0.5
        
        # Food and poison properties
        self.food_radius = 8
        self.poison_radius = 8
        self.food_positions = []
        self.poison_positions = []
        self.food_velocities = []
        self.poison_velocities = []
        
        # Define action and observation spaces
        # Action: [force_x, force_y] - continuous values between -1 and 1
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # Observation: [agent_pos_x, agent_pos_y, agent_vel_x, agent_vel_y] + 
        #              [sensor_1, sensor_2, ..., sensor_n] for both food and poison
        # Each sensor returns the distance to the closest object in that direction
        obs_dim = 4 + self.num_sensors * 2  # Position, velocity, and sensor readings
        self.observation_space = spaces.Box(
            low=-float('inf'), high=float('inf'), shape=(obs_dim,), dtype=np.float32
        )
    
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
        super().reset(seed=seed)
        self.steps = 0
        
        # Reset agent
        self.agent_pos = np.array([self.width / 2, self.height / 2], dtype=np.float32)
        self.agent_vel = np.zeros(2, dtype=np.float32)
        
        # Reset food and poison
        self.food_positions = []
        self.poison_positions = []
        self.food_velocities = []
        self.poison_velocities = []
        
        # Generate random positions for food and poison
        for _ in range(self.num_food):
            pos = np.array([
                np.random.uniform(self.food_radius, self.width - self.food_radius),
                np.random.uniform(self.food_radius, self.height - self.food_radius)
            ], dtype=np.float32)
            vel = np.random.uniform(-0.5, 0.5, 2)
            self.food_positions.append(pos)
            self.food_velocities.append(vel)
        
        for _ in range(self.num_poison):
            pos = np.array([
                np.random.uniform(self.poison_radius, self.width - self.poison_radius),
                np.random.uniform(self.poison_radius, self.height - self.poison_radius)
            ], dtype=np.float32)
            vel = np.random.uniform(-0.5, 0.5, 2)
            self.poison_positions.append(pos)
            self.poison_velocities.append(vel)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: The action to take [force_x, force_y]
            
        Returns:
            observation: The new observation
            reward: The reward for the action
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated (e.g., due to max steps)
            info: Additional information
        """
        self.steps += 1
        
        # Apply action (force) to agent
        force = np.clip(action, -1.0, 1.0) * self.max_force
        self.agent_vel += force
        
        # Apply friction
        self.agent_vel *= (1 - self.friction)
        
        # Limit speed
        speed = np.linalg.norm(self.agent_vel)
        if speed > self.max_speed:
            self.agent_vel = self.agent_vel / speed * self.max_speed
        
        # Update agent position
        self.agent_pos += self.agent_vel
        
        # Boundary conditions (bounce off walls)
        if self.agent_pos[0] < self.agent_radius:
            self.agent_pos[0] = self.agent_radius
            self.agent_vel[0] *= -0.5  # Bounce with energy loss
        elif self.agent_pos[0] > self.width - self.agent_radius:
            self.agent_pos[0] = self.width - self.agent_radius
            self.agent_vel[0] *= -0.5
        
        if self.agent_pos[1] < self.agent_radius:
            self.agent_pos[1] = self.agent_radius
            self.agent_vel[1] *= -0.5
        elif self.agent_pos[1] > self.height - self.agent_radius:
            self.agent_pos[1] = self.height - self.agent_radius
            self.agent_vel[1] *= -0.5
        
        # Update food and poison positions
        reward = 0
        food_to_remove = []
        
        for i, (pos, vel) in enumerate(zip(self.food_positions, self.food_velocities)):
            # Update position
            pos += vel
            
            # Boundary conditions
            if pos[0] < self.food_radius or pos[0] > self.width - self.food_radius:
                vel[0] *= -1
            if pos[1] < self.food_radius or pos[1] > self.height - self.food_radius:
                vel[1] *= -1
            
            # Check collision with agent
            dist = np.linalg.norm(pos - self.agent_pos)
            if dist < self.agent_radius + self.food_radius:
                reward += 1.0
                food_to_remove.append(i)
            
            # Update position and velocity
            self.food_positions[i] = pos
            self.food_velocities[i] = vel
        
        # Remove eaten food
        for i in sorted(food_to_remove, reverse=True):
            del self.food_positions[i]
            del self.food_velocities[i]
            
            # Add new food
            new_pos = np.array([
                np.random.uniform(self.food_radius, self.width - self.food_radius),
                np.random.uniform(self.food_radius, self.height - self.food_radius)
            ], dtype=np.float32)
            new_vel = np.random.uniform(-0.5, 0.5, 2)
            self.food_positions.append(new_pos)
            self.food_velocities.append(new_vel)
        
        # Update poison positions
        poison_to_remove = []
        
        for i, (pos, vel) in enumerate(zip(self.poison_positions, self.poison_velocities)):
            # Update position
            pos += vel
            
            # Boundary conditions
            if pos[0] < self.poison_radius or pos[0] > self.width - self.poison_radius:
                vel[0] *= -1
            if pos[1] < self.poison_radius or pos[1] > self.height - self.poison_radius:
                vel[1] *= -1
            
            # Check collision with agent
            dist = np.linalg.norm(pos - self.agent_pos)
            if dist < self.agent_radius + self.poison_radius:
                reward -= 1.0
                poison_to_remove.append(i)
            
            # Update position and velocity
            self.poison_positions[i] = pos
            self.poison_velocities[i] = vel
        
        # Remove consumed poison
        for i in sorted(poison_to_remove, reverse=True):
            del self.poison_positions[i]
            del self.poison_velocities[i]
            
            # Add new poison
            new_pos = np.array([
                np.random.uniform(self.poison_radius, self.width - self.poison_radius),
                np.random.uniform(self.poison_radius, self.height - self.poison_radius)
            ], dtype=np.float32)
            new_vel = np.random.uniform(-0.5, 0.5, 2)
            self.poison_positions.append(new_pos)
            self.poison_velocities.append(new_vel)
        
        # Small negative reward for each step to encourage efficiency
        reward -= 0.01
        
        # Check if episode is done
        terminated = False
        truncated = self.steps >= self.max_steps
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        """
        Get the current observation of the environment.
        
        Returns:
            observation: The current observation
        """
        # Agent position and velocity
        agent_state = np.concatenate([self.agent_pos, self.agent_vel])
        
        # Sensor readings
        food_readings = np.ones(self.num_sensors) * self.sensor_range
        poison_readings = np.ones(self.num_sensors) * self.sensor_range
        
        # Calculate sensor directions
        sensor_angles = np.linspace(0, 2 * np.pi, self.num_sensors, endpoint=False)
        sensor_dirs = np.array([[np.cos(angle), np.sin(angle)] for angle in sensor_angles])
        
        # Check food sensors
        for food_pos in self.food_positions:
            rel_pos = food_pos - self.agent_pos
            dist = np.linalg.norm(rel_pos)
            
            if dist <= self.sensor_range:
                # Calculate angle to food
                angle = np.arctan2(rel_pos[1], rel_pos[0])
                if angle < 0:
                    angle += 2 * np.pi
                
                # Find closest sensor
                sensor_idx = int((angle / (2 * np.pi)) * self.num_sensors) % self.num_sensors
                
                # Update sensor reading if closer than current reading
                if dist < food_readings[sensor_idx]:
                    food_readings[sensor_idx] = dist
        
        # Check poison sensors
        for poison_pos in self.poison_positions:
            rel_pos = poison_pos - self.agent_pos
            dist = np.linalg.norm(rel_pos)
            
            if dist <= self.sensor_range:
                # Calculate angle to poison
                angle = np.arctan2(rel_pos[1], rel_pos[0])
                if angle < 0:
                    angle += 2 * np.pi
                
                # Find closest sensor
                sensor_idx = int((angle / (2 * np.pi)) * self.num_sensors) % self.num_sensors
                
                # Update sensor reading if closer than current reading
                if dist < poison_readings[sensor_idx]:
                    poison_readings[sensor_idx] = dist
        
        # Normalize sensor readings
        food_readings = food_readings / self.sensor_range
        poison_readings = poison_readings / self.sensor_range
        
        # Combine all observations
        observation = np.concatenate([agent_state, food_readings, poison_readings])
        
        return observation
    
    def _get_info(self):
        """
        Get additional information about the environment.
        
        Returns:
            info: A dictionary of additional information
        """
        return {
            "agent_position": self.agent_pos.copy(),
            "agent_velocity": self.agent_vel.copy(),
            "food_count": len(self.food_positions),
            "poison_count": len(self.poison_positions),
            "steps": self.steps
        }
    
    def _render_environment(self, canvas):
        """
        Render environment-specific elements to the canvas.
        
        Args:
            canvas: Pygame surface to draw on
        """
        # Draw food (green)
        for pos in self.food_positions:
            pygame.draw.circle(
                canvas,
                (0, 255, 0),
                pos.astype(int),
                self.food_radius
            )
        
        # Draw poison (red)
        for pos in self.poison_positions:
            pygame.draw.circle(
                canvas,
                (255, 0, 0),
                pos.astype(int),
                self.poison_radius
            )
        
        # Draw agent (blue)
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            self.agent_pos.astype(int),
            self.agent_radius
        )
        
        # Draw agent direction (velocity)
        if np.linalg.norm(self.agent_vel) > 0:
            vel_normalized = self.agent_vel / np.linalg.norm(self.agent_vel)
            end_pos = self.agent_pos + vel_normalized * self.agent_radius
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                self.agent_pos.astype(int),
                end_pos.astype(int),
                2
            )
        
        # Draw sensor range (optional, for debugging)
        if False:  # Set to True to visualize sensors
            pygame.draw.circle(
                canvas,
                (200, 200, 200),
                self.agent_pos.astype(int),
                self.sensor_range,
                1
            )
            
            # Draw sensor directions
            sensor_angles = np.linspace(0, 2 * np.pi, self.num_sensors, endpoint=False)
            for angle in sensor_angles:
                end_x = self.agent_pos[0] + np.cos(angle) * self.sensor_range
                end_y = self.agent_pos[1] + np.sin(angle) * self.sensor_range
                pygame.draw.line(
                    canvas,
                    (200, 200, 200),
                    self.agent_pos.astype(int),
                    (int(end_x), int(end_y)),
                    1
                ) 