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

Generated with Cursor Sonnet 3.7 from the following prompt:

```


YouTube
Here's a summary of the video, focusing on the key points:

Limitations of Generative AI: Current AI excels at text-based tasks but struggles with real-world actions, decision-making, and trial-and-error learning [00:38].
Need for Action-Oriented AI: To move beyond supervised learning, AI systems need to take actions and learn from experience [00:53]. This requires training in simulated environments, creating an "internet of environments" [01:29].
Compute-Only Scaling: As human data is limited, relying on increasing computing power and simulated environments is crucial for AI advancements [02:01].
Challenge of Environment Design: Designing task distributions that allow agents to be robust to various real-world scenarios is a key challenge [02:35].
Regret and Learnability: The video discusses the concept of "regret" (difference between agent performance and optimal performance) and its limitations in complex environments [04:04]. The speaker introduces "learnability" (where an agent sometimes succeeds) as a more effective metric for training [08:08].
Multi-Agent Systems: The research moved from single-agent grid worlds to multi-agent systems with continuous action and state spaces, simulating robots with lidar [06:16].
GPU Acceleration: To address the computational cost of running experiments, the lab developed "RL at the Hyperscale," which puts the environment and policy training loop on the GPU, resulting in significant speedups [15:02].
Kinetics: A Mini-Universe for AI Training: The video introduces Kinetics, a 2D physics simulator that allows for diverse task creation and GPU-accelerated training [20:55]. This environment enables the training of generalist agents that show zero-shot improvement and faster fine-tuning on new tasks [23:39].
Agentic Foundation Models: The goal is to create foundation models for decision-making and acting, similar to how large language models are pre-trained for text-based tasks [26:20].
Future Directions: The speaker suggests that the current work in 2D environments can be scaled up to 3D and more complex settings with increased computational power [26:39].
Would you like to explore other aspects of this video or perhaps find other related videos?






this too https://youtu.be/1kwbp8hRRfs?si=cqfO16p_RO6DKDzT



YouTube
Here's a summary of the video, focusing on the key points:

Introduction of AI-driven algorithm discovery [00:04]
The increasing capabilities of AI, particularly Large Language Models (LLMs), are being explored for generating novel ideas and optimizing algorithms.
LLMs can potentially act as "AI scientists" to generate data and discover deeper knowledge than what is currently published. [00:14]
Discovering Preference Optimization Algorithms [02:16]
The video discusses a paper on discovering preference optimization algorithms using LLMs.
Instead of handcrafting algorithms, the idea is to use LLMs to optimize the algorithms used to train language models. [03:05]
LLMs can explore a broader range of algorithms due to their ability to perform more trials and mix different concepts from various fields. [03:45]
How the Approach Works [03:51]
LLMs are used to propose code snippets for objective functions, providing names and rationales for their ideas. [04:28]
The generated code is evaluated, and the results are fed back into the LLM to refine and improve the objective functions. [04:42]
LLMs excel at combining concepts from diverse fields like chemistry, physics, and machine learning to discover new algorithms. [05:15]
LLMs as Mutation Operators [05:37]
LLMs are considered strong and intelligent mutation operators that can aid in discovering new algorithms.
Their ability to leverage pre-trained code knowledge allows for more efficient exploration of the algorithm space. [06:19]
Limitations and Future Directions [08:07]
The discussion touches on the limitations of LLMs being bounded by their training distribution and the need for creativity beyond interpolation.
There is potential to expand the optimization criteria to include fairness, bias, and other important factors beyond single-number metrics. [09:22]
The Role of Humans [17:05]
The video explores the role of humans in supervising AI scientists, suggesting a collaborative approach where humans provide guidance and taste-making.
Infinite Regress and Model Collapse [18:47]
The possibility of an infinite regress is considered, where improved loss functions are used to train new models, potentially leading to mode collapse.
The importance of entropy and diverse inputs in preventing model collapse is highlighted. [20:31]
LLMs as Evolutionary Strategies [25:33]
The video transitions to discussing LLMs as evolutionary strategies for blackbox optimization.
LLMs can optimize numerical blackbox optimization problems by applying in-context learning to abstract representations. [26:24]
Automated Design of Agentic Systems [48:10]
The discussion shifts to the automated design of agentic systems using LLMs.
LLMs are now capable of writing code for complex agent systems, leading to the possibility of LLM agents writing other LLM agents. [49:14]
Intelligent Go Explore [01:09:21]
The video introduces the "Intelligent Go Explore" algorithm, which uses LLMs to guide exploration in reinforcement learning environments.
LLMs' inherent understanding of human-like interestingness is leveraged to discover new states in complex environments. [01:10:05]
AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery [01:19:53]
The video concludes by discussing the "AI Scientist" paper, which explores the use of LLMs to fully automate scientific discovery.
The system aims to generate new papers by chaining together ideation, coding, experimentation, and paper writing. [01:20:17]


```