import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
import pygame

def plot_scores(scores, window_size=100):
    """
    Plot the scores from training.
    
    Args:
        scores (list): List of scores from each episode
        window_size (int): Window size for moving average
    """
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(scores)), scores, alpha=0.3, label='Score')
    
    # Plot moving average
    if len(scores) >= window_size:
        moving_avg = np.convolve(scores, np.ones(window_size) / window_size, mode='valid')
        plt.plot(np.arange(len(moving_avg)) + window_size - 1, moving_avg, label=f'Moving Avg (window={window_size})')
    
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Scores')
    plt.legend()
    plt.grid(True)
    
    return plt.gcf()

def save_training_plot(scores, filename, window_size=100):
    """
    Save a plot of training scores.
    
    Args:
        scores (list): List of scores from each episode
        filename (str): Path to save the plot
        window_size (int): Window size for moving average
    """
    fig = plot_scores(scores, window_size)
    fig.savefig(filename)
    plt.close(fig)

def create_episode_animation(env, agent, max_steps=1000, interval=50):
    """
    Create an animation of an episode.
    
    Args:
        env: Environment to run the episode in
        agent: Agent to use for actions
        max_steps (int): Maximum number of steps in the episode
        interval (int): Interval between frames in milliseconds
        
    Returns:
        matplotlib.animation.FuncAnimation: Animation of the episode
    """
    # Reset environment
    state, _ = env.reset()
    frames = []
    
    # Run episode
    for _ in range(max_steps):
        # Render frame
        frame = env.render()
        frames.append(frame)
        
        # Take action
        action = agent.select_action(state)
        state, _, done, truncated, _ = env.step(action)
        
        if done or truncated:
            break
    
    # Create animation
    fig = plt.figure(figsize=(8, 8))
    plt.axis('off')
    
    ims = []
    for frame in frames:
        im = plt.imshow(frame, animated=True)
        plt.axis('off')
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True)
    
    return ani

def save_episode_animation(env, agent, filename, max_steps=1000, fps=30):
    """
    Save an animation of an episode.
    
    Args:
        env: Environment to run the episode in
        agent: Agent to use for actions
        filename (str): Path to save the animation
        max_steps (int): Maximum number of steps in the episode
        fps (int): Frames per second
    """
    ani = create_episode_animation(env, agent, max_steps, 1000 // fps)
    ani.save(filename, fps=fps)
    plt.close()

def visualize_q_values(agent, state, discretization=5):
    """
    Visualize the Q-values for a given state.
    
    Args:
        agent: DQN agent
        state: State to visualize Q-values for
        discretization (int): Number of discrete actions per dimension
    """
    # Convert state to tensor
    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(agent.device)
    
    # Get Q-values
    agent.qnetwork_local.eval()
    with torch.no_grad():
        q_values = agent.qnetwork_local(state_tensor).cpu().numpy()[0]
    agent.qnetwork_local.train()
    
    # Reshape Q-values for 2D action space
    if agent.action_size == 2:
        q_values_2d = q_values.reshape(discretization, discretization)
        
        # Plot Q-values
        plt.figure(figsize=(8, 6))
        plt.imshow(q_values_2d, cmap='viridis')
        plt.colorbar(label='Q-value')
        plt.title('Q-values for Current State')
        plt.xlabel('Action Dimension 1')
        plt.ylabel('Action Dimension 2')
        
        # Add text annotations
        for i in range(discretization):
            for j in range(discretization):
                plt.text(j, i, f'{q_values_2d[i, j]:.2f}', 
                         ha='center', va='center', color='white')
        
        return plt.gcf()
    else:
        # For non-2D action spaces, just plot as a bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(q_values)), q_values)
        plt.xlabel('Action Index')
        plt.ylabel('Q-value')
        plt.title('Q-values for Current State')
        
        return plt.gcf()

def save_q_value_plot(agent, state, filename, discretization=5):
    """
    Save a plot of Q-values for a given state.
    
    Args:
        agent: DQN agent
        state: State to visualize Q-values for
        filename (str): Path to save the plot
        discretization (int): Number of discrete actions per dimension
    """
    fig = visualize_q_values(agent, state, discretization)
    fig.savefig(filename)
    plt.close(fig)

def record_video(env, agent, filename, max_steps=1000, fps=30):
    """
    Record a video of an episode.
    
    Args:
        env: Environment to run the episode in
        agent: Agent to use for actions
        filename (str): Path to save the video
        max_steps (int): Maximum number of steps in the episode
        fps (int): Frames per second
    """
    # Reset environment
    state, _ = env.reset()
    frames = []
    
    # Run episode
    for _ in range(max_steps):
        # Render frame
        frame = env.render()
        frames.append(frame)
        
        # Take action
        action = agent.select_action(state)
        state, _, done, truncated, _ = env.step(action)
        
        if done or truncated:
            break
    
    # Save video using matplotlib animation
    fig = plt.figure(figsize=(8, 8))
    plt.axis('off')
    
    ims = []
    for frame in frames:
        im = plt.imshow(frame, animated=True)
        plt.axis('off')
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=1000 // fps, blit=True)
    ani.save(filename, fps=fps)
    plt.close() 