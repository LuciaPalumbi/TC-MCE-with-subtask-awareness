import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import torch

def trajectory_reward_visualization(theta_list, trajectory, filename, feat):
    print("Generating visualizations...")
    rewards = []
    subtasks = []
    for state, _ in trajectory:
        feature_vector = torch.tensor(feat.feature_vector(state), dtype=torch.float32)
        subtask = int(state[-2])
        subtasks.append(subtask)
        theta = theta_list[subtask]
        if theta is not None:
            feature_vector = feature_vector.to(theta.device)
            reward = torch.dot(feature_vector, theta).item()
            rewards.append(reward)

    cumulative_reward = sum(rewards)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot rewards with color changes for subtasks
    colors = ['blue', 'orange', 'green']
    start = 0
    for i in range(1, len(subtasks)):
        if subtasks[i] != subtasks[i-1] or i == len(subtasks) - 1:
            end = i if i < len(subtasks) - 1 else len(subtasks)
            ax.plot(range(start, end), rewards[start:end], color=colors[subtasks[start]], label=f'Subtask {subtasks[start]}' if start == 0 else "")
            start = i

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Reward')
    ax.set_title(f'Reward along trajectory: {filename}\nCumulative Reward: {cumulative_reward:.2f}')

    # Color background based on subtasks
    subtask_changes = np.where(np.diff(subtasks) != 0)[0]
    subtask_changes = np.concatenate(([0], subtask_changes + 1, [len(subtasks)]))
    bg_colors = ['lightblue', 'lightorange', 'lightgreen']
    for i in range(len(subtask_changes) - 1):
        ax.axvspan(subtask_changes[i], subtask_changes[i+1], facecolor=bg_colors[subtasks[subtask_changes[i]]], alpha=0.3)

    ax.legend()
    plt.tight_layout()
    plt.show()

    return cumulative_reward

def cumulative_rewards_comparison(trajectories, theta_list, feat):
    cumulative_rewards = [[] for _ in range(len(theta_list))]
    for filename, trajectory in trajectories.items():
        rewards = trajectory_reward_visualization(theta_list, trajectory, filename, feat)
        for i, reward in enumerate(rewards):
            if theta_list[i] is not None:
                cumulative_rewards[i].append((filename, reward))

    fig, axes = plt.subplots(len(theta_list), 1, figsize=(12, 6*len(theta_list)), sharex=True)
    if len(theta_list) == 1:
        axes = [axes]

    for i, rewards in enumerate(cumulative_rewards):
        if rewards:
            rewards.sort(key=lambda x: x[1], reverse=True)
            filenames, reward_values = zip(*rewards)
            ax = axes[i]
            bars = ax.bar(filenames, reward_values)
            ax.axhline(y=np.mean(reward_values), color='r', linestyle='--', label='Mean Reward')
            ax.set_ylabel('Cumulative Reward')
            ax.set_title(f'Cumulative Rewards Comparison - Subtask {i}')
            ax.legend()
            ax.tick_params(axis='x', rotation=90)

            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                        f'{bar.get_height():.2f}', ha='center', va='bottom')

    plt.xlabel('Trajectory')
    plt.tight_layout()
    plt.show()

def feature_value_visualization(feat):
    """
    Visualize feature values distribution.
    """
    n_features = feat.num_features
    n_bins = feat.n_bins

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    for i in range(n_features - 1):  # Exclude subtask feature
        feature_values = np.linspace(0, 1, n_bins)  # Assuming normalized feature values
        ax = axes[i]
        sns.barplot(x=feature_values, y=np.ones(n_bins), ax=ax)
        ax.set_title(f'Feature {i+1}')
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('Count')

    # Subtask distribution
    subtask_counts = [0, 0, 0]  # Placeholder, you should compute actual counts
    ax = axes[-1]
    sns.barplot(x=['Subtask 0', 'Subtask 1', 'Subtask 2'], y=subtask_counts, ax=ax)
    ax.set_title('Subtask Distribution')
    ax.set_ylabel('Count')

    plt.tight_layout()
    plt.show()

def state_visitation_frequency(trajectories, feat):
    """
    Create a heatmap of state visitation frequency.
    """
    state_counts = {}
    for trajectory in trajectories.values():
        for state, _ in trajectory:
            state_tuple = tuple(feat.feature_vector(state)[:-1])  # Exclude subtask
            if state_tuple in state_counts:
                state_counts[state_tuple] += 1
            else:
                state_counts[state_tuple] = 1

    # Convert to 2D array for heatmap
    max_counts = max(state_counts.values())
    heatmap_data = np.zeros((feat.n_bins, feat.n_bins))
    for state, count in state_counts.items():
        heatmap_data[int(state[0]*feat.n_bins), int(state[1]*feat.n_bins)] = count / max_counts

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap='YlOrRd')
    plt.title('State Visitation Frequency')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def policy_visualization(policy, feat):
    """
    Create a quiver plot to visualize the policy.
    """
    n_bins = feat.n_bins
    x = np.linspace(0, 1, n_bins)
    y = np.linspace(0, 1, n_bins)
    X, Y = np.meshgrid(x, y)

    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(n_bins):
        for j in range(n_bins):
            action_probs = policy[i*n_bins + j]
            best_action = torch.argmax(action_probs).item()
            U[i, j] = np.cos(best_action * 2 * np.pi / len(action_probs))
            V[i, j] = np.sin(best_action * 2 * np.pi / len(action_probs))

    plt.figure(figsize=(10, 8))
    plt.quiver(X, Y, U, V)
    plt.title('Policy Visualization')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def learning_curve_visualization(delta_history):
    """
    Plot the learning curve (delta over iterations).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(delta_history)
    plt.title('Learning Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Delta (Max change in theta)')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

# Main visualization function
def visualize_results(trajectories, theta_list, feat, policies, delta_history):
    print("Generating visualizations...")
    
    for subtask, (theta, policy) in enumerate(zip(theta_list, policies)):
        if theta is not None and policy is not None:
            print(f"Visualizing results for subtask {subtask}")

            # Trajectory reward visualization for a sample trajectory
            sample_filename = next(iter(trajectories))
            sample_trajectory = trajectories[sample_filename]
            trajectory_reward_visualization(theta, sample_trajectory, sample_filename, feat)

            # Cumulative rewards comparison
            cumulative_rewards_comparison(trajectories, theta, feat)

            # Feature value visualization
            feature_value_visualization(feat)

            # State visitation frequency
            state_visitation_frequency(trajectories, feat)

            # Policy visualization
            policy_visualization(policy, feat)

            # Learning curve visualization
            learning_curve_visualization(delta_history)

    print("Visualization complete.")