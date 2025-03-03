import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple

# Define a named tuple for storing experiences
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    """
    Replay buffer for storing and sampling experiences.
    """
    
    def __init__(self, capacity):
        """
        Initialize the replay buffer.
        
        Args:
            capacity (int): Maximum capacity of the buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, next_state, reward, done):
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            reward: Reward received
            done: Whether the episode is done
        """
        experience = Experience(state, action, next_state, reward, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size (int): Size of the batch to sample
            
        Returns:
            Tuple of (states, actions, next_states, rewards, dones)
        """
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float()
        
        return states, actions, next_states, rewards, dones
    
    def __len__(self):
        """
        Get the current size of the buffer.
        
        Returns:
            int: Current size of the buffer
        """
        return len(self.buffer)

class QNetwork(nn.Module):
    """
    Neural network for approximating the Q-function.
    """
    
    def __init__(self, state_size, action_size, hidden_size=64):
        """
        Initialize the Q-network.
        
        Args:
            state_size (int): Dimension of the state space
            action_size (int): Dimension of the action space
            hidden_size (int): Size of the hidden layers
        """
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: Input state
            
        Returns:
            Q-values for each action
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """
    Deep Q-Network agent for continuous action spaces.
    
    This implementation uses a discretized action space for simplicity.
    """
    
    def __init__(self, observation_space, action_space, hidden_size=64, learning_rate=1e-3,
                 buffer_size=10000, batch_size=64, gamma=0.99, tau=1e-3, update_every=4,
                 discretization=5):
        """
        Initialize the DQN agent.
        
        Args:
            observation_space: Observation space from the environment
            action_space: Action space from the environment
            hidden_size (int): Size of the hidden layers in the Q-network
            learning_rate (float): Learning rate for the optimizer
            buffer_size (int): Size of the replay buffer
            batch_size (int): Batch size for training
            gamma (float): Discount factor
            tau (float): Soft update parameter
            update_every (int): How often to update the target network
            discretization (int): Number of discrete actions per dimension
        """
        self.state_size = observation_space.shape[0]
        self.action_size = action_space.shape[0]
        self.discretization = discretization
        self.discrete_action_size = discretization ** self.action_size
        
        # Create action mapping
        self.action_map = self._create_action_map(discretization, action_space.low, action_space.high)
        
        # Q-Networks
        self.qnetwork_local = QNetwork(self.state_size, self.discrete_action_size, hidden_size)
        self.qnetwork_target = QNetwork(self.state_size, self.discrete_action_size, hidden_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        
        # Learning parameters
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.t_step = 0
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.qnetwork_local.to(self.device)
        self.qnetwork_target.to(self.device)
    
    def _create_action_map(self, discretization, low, high):
        """
        Create a mapping from discrete action indices to continuous actions.
        
        Args:
            discretization (int): Number of discrete actions per dimension
            low: Lower bound of the action space
            high: Upper bound of the action space
            
        Returns:
            List of continuous actions
        """
        # For 2D action space with discretization=3, this creates a 3x3 grid of actions
        # e.g., [(-1,-1), (-1,0), (-1,1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1)]
        action_map = []
        
        # Create all combinations of discrete actions
        for i in range(self.discrete_action_size):
            action = []
            temp = i
            
            for dim in range(self.action_size):
                action.append(temp % discretization)
                temp //= discretization
            
            # Convert to continuous action
            continuous_action = []
            for dim, val in enumerate(action):
                # Map from [0, discretization-1] to [low, high]
                continuous_val = low[dim] + (val / (discretization - 1)) * (high[dim] - low[dim])
                continuous_action.append(continuous_val)
            
            action_map.append(np.array(continuous_action, dtype=np.float32))
        
        return action_map
    
    def step(self, state, action, reward, next_state, done):
        """
        Update the agent's knowledge based on the new experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Convert continuous action to discrete index
        action_idx = self._continuous_to_discrete(action)
        
        # Add experience to replay buffer
        self.memory.add(state, action, next_state, reward, done)
        
        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self._learn(experiences)
    
    def _continuous_to_discrete(self, continuous_action):
        """
        Convert a continuous action to its closest discrete index.
        
        Args:
            continuous_action: Continuous action
            
        Returns:
            int: Index of the closest discrete action
        """
        min_dist = float('inf')
        closest_idx = 0
        
        for i, action in enumerate(self.action_map):
            dist = np.sum((continuous_action - action) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        return closest_idx
    
    def select_action(self, state, eps=None):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            eps (float, optional): Epsilon value for exploration
            
        Returns:
            numpy.ndarray: Selected action
        """
        if eps is None:
            eps = self.epsilon
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            
            # Get the index of the action with the highest Q-value
            action_idx = torch.argmax(action_values).item()
            
            # Convert to continuous action
            return self.action_map[action_idx]
        else:
            # Random action
            return self.action_map[random.randrange(self.discrete_action_size)]
    
    def _learn(self, experiences):
        """
        Update the Q-networks based on a batch of experiences.
        
        Args:
            experiences: Tuple of (states, actions, next_states, rewards, dones)
        """
        states, actions, next_states, rewards, dones = experiences
        
        # Convert actions to indices
        action_indices = torch.zeros(actions.shape[0], dtype=torch.int64)
        for i in range(actions.shape[0]):
            action_indices[i] = self._continuous_to_discrete(actions[i].numpy())
        
        # Get max predicted Q values for next states from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, action_indices.unsqueeze(1))
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self._soft_update(self.qnetwork_local, self.qnetwork_target)
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _soft_update(self, local_model, target_model):
        """
        Soft update of the target network parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Args:
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def train(self, env, episodes=1000, max_steps=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995,
              print_every=100):
        """
        Train the agent on the given environment.
        
        Args:
            env: Environment to train on
            episodes (int): Number of episodes to train
            max_steps (int): Maximum number of steps per episode
            eps_start (float): Starting value of epsilon
            eps_end (float): Minimum value of epsilon
            eps_decay (float): Decay rate of epsilon
            print_every (int): How often to print progress
            
        Returns:
            list: Scores for each episode
        """
        scores = []
        eps = eps_start
        
        for i_episode in range(1, episodes + 1):
            state, _ = env.reset()
            score = 0
            
            for t in range(max_steps):
                action = self.select_action(state, eps)
                next_state, reward, done, truncated, _ = env.step(action)
                self.step(state, action, reward, next_state, done or truncated)
                state = next_state
                score += reward
                
                if done or truncated:
                    break
            
            scores.append(score)
            eps = max(eps_end, eps_decay * eps)
            
            if i_episode % print_every == 0:
                print(f'Episode {i_episode}\tAverage Score: {np.mean(scores[-print_every:]):.2f}')
        
        return scores
    
    def save(self, filename):
        """
        Save the agent's model.
        
        Args:
            filename (str): Path to save the model
        """
        torch.save({
            'qnetwork_local_state_dict': self.qnetwork_local.state_dict(),
            'qnetwork_target_state_dict': self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
    
    def load(self, filename):
        """
        Load the agent's model.
        
        Args:
            filename (str): Path to load the model from
        """
        checkpoint = torch.load(filename)
        self.qnetwork_local.load_state_dict(checkpoint['qnetwork_local_state_dict'])
        self.qnetwork_target.load_state_dict(checkpoint['qnetwork_target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon'] 