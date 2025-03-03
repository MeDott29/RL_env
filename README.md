# Sakana: A Simple Physics-Based Reinforcement Learning Environment

Sakana is a lightweight 2D physics-based environment for training and evaluating reinforcement learning agents. Inspired by research in AI agent training environments like those discussed in Anthropic's work on "Kinetics," this project aims to provide a simple but extensible platform for experimenting with RL algorithms.

## Features

- 2D physics-based environment with customizable parameters
- Support for both single-agent and multi-agent scenarios
- GPU-accelerated training capabilities (via PyTorch)
- Visualization tools for monitoring agent performance
- Extensible framework for creating new tasks and environments

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sakana.git
cd sakana

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from sakana.environments import WaterWorld
from sakana.agents import DQNAgent

# Create environment
env = WaterWorld(width=400, height=400)

# Create agent
agent = DQNAgent(env.observation_space, env.action_space)

# Train the agent
agent.train(env, episodes=1000)

# Visualize results
env.render_episode(agent)
```

## Project Structure

- `sakana/environments/`: Contains environment implementations
- `sakana/agents/`: Contains agent implementations
- `sakana/utils/`: Utility functions and visualization tools

## License

MIT 