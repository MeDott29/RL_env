import numpy as np
import matplotlib.pyplot as plt
import os
import time

from sakana.environments import WaterWorld
from sakana.agents import DQNAgent
from sakana.utils import (
    plot_scores,
    save_training_plot,
    save_episode_animation,
    visualize_q_values
)

def main():
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Create environment
    env = WaterWorld(
        width=600,
        height=600,
        render_mode='rgb_array',
        num_food=10,
        num_poison=10,
        sensor_range=150,
        friction=0.05,
        max_steps=1000
    )
    
    # Create agent
    agent = DQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hidden_size=128,
        learning_rate=1e-3,
        buffer_size=100000,
        batch_size=64,
        gamma=0.99,
        tau=1e-3,
        update_every=4,
        discretization=5
    )
    
    # Train agent (short training for demonstration)
    print("Training agent...")
    start_time = time.time()
    scores = agent.train(
        env=env,
        episodes=100,  # Increase for better performance
        max_steps=500,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        print_every=10
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save training plot
    save_training_plot(scores, 'output/training_scores.png', window_size=10)
    print("Training plot saved to output/training_scores.png")
    
    # Save model
    agent.save('output/dqn_agent.pth')
    print("Agent model saved to output/dqn_agent.pth")
    
    # Create a new environment for rendering
    render_env = WaterWorld(
        width=600,
        height=600,
        render_mode='rgb_array',
        num_food=10,
        num_poison=10,
        sensor_range=150,
        friction=0.05,
        max_steps=1000
    )
    
    # Save an episode animation
    print("Creating episode animation...")
    save_episode_animation(
        env=render_env,
        agent=agent,
        filename='output/episode.mp4',
        max_steps=500,
        fps=30
    )
    print("Episode animation saved to output/episode.mp4")
    
    # Visualize Q-values for a sample state
    state, _ = render_env.reset()
    q_value_fig = visualize_q_values(agent, state, discretization=5)
    q_value_fig.savefig('output/q_values.png')
    plt.close(q_value_fig)
    print("Q-value visualization saved to output/q_values.png")
    
    # Run an interactive episode if pygame is available
    try:
        print("\nRunning interactive episode...")
        interactive_env = WaterWorld(
            width=600,
            height=600,
            render_mode='human',
            num_food=10,
            num_poison=10,
            sensor_range=150,
            friction=0.05,
            max_steps=1000
        )
        
        interactive_env.render_episode(agent, max_steps=500, delay=0.01)
    except Exception as e:
        print(f"Could not run interactive episode: {e}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main() 