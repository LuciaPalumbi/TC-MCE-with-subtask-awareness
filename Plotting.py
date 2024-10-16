#!/usr/bin/env python3

import numpy as np
import torch
import features as F
import mdp as MDP
import data_manip as DM
from datetime import datetime
from random_trajectory_generator import generate_random_trajectories
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# for visualisation
#######################################################################################################################
def plot_weights(theta):
    theta = theta.cpu()
    theta_np = theta.detach().numpy()

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig1.canvas.manager.set_window_title("Feature Weights")
    # Figure 1: Feature Weights (Theta)
    ax1.bar(range(len(theta_np)), theta_np)
    ax1.set_title('Feature Weights (Theta)')
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('Weight Value')
    plt.tight_layout()

def visualize_reward_function(feature_matrix, theta, positions, velocities, forces):
    # Ensure both tensors are on the same device (CPU in this case)
    feature_matrix = feature_matrix.cpu()
    theta = theta.cpu()
    
    # Calculate the learned reward for each state
    state_reward_vector = feature_matrix @ theta        # NsxNf * Nfx1 = Nsx1
    
    # Convert to numpy arrays for plotting
    state_reward_vector = state_reward_vector.detach().numpy()
    theta_np = theta.detach().numpy()
    
    n_states, n_features = feature_matrix.shape
    n_pos = len(positions)
    n_vel = len(velocities)
    n_force = len(forces)
    
    # Create two figures
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig1.canvas.manager.set_window_title("Feature Weights")
    fig2, (ax2, ax3, ax4) = plt.subplots(1, 3, figsize=(20, 6))
    fig2.canvas.manager.set_window_title("Variables Importance")
    
    # Figure 1: Feature Weights (Theta)
    ax1.bar(range(len(theta_np)), theta_np)
    ax1.set_title('Feature Weights (Theta)')
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('Weight Value')
    
    # Figure 2: Position Feature Importance
    ax2.bar(positions, theta_np[:n_pos])
    ax2.set_title('Position Feature Importance')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Weight Value')
    
    # Figure 2: Velocity Feature Importance
    ax3.bar(velocities, theta_np[n_pos:n_pos+n_vel])
    ax3.set_title('Velocity Feature Importance')
    ax3.set_xlabel('Velocity')
    ax3.set_ylabel('Weight Value')
    
    # Figure 2: Force Feature Importance
    ax4.bar(forces, theta_np[n_pos+n_vel:n_pos+n_vel+n_force])
    ax4.set_title('Force Feature Importance')
    ax4.set_xlabel('Force')
    ax4.set_ylabel('Weight Value')
    
    plt.tight_layout()
    #plt.show()

def plot_trajectory_rewards(trajectories, state_to_index, feature_matrix, theta):

    feature_matrix = feature_matrix.cpu() 
    theta = theta.cpu()

    # Calculate the learned reward for each state
    state_reward_vector = feature_matrix @ theta        # NsxNf * Nfx1 = Nsx1

    # Convert to numpy arrays for plotting
    state_reward_vector = state_reward_vector.detach().numpy()

    num_trajectories = len(trajectories)
    cols = 4
    rows = math.ceil(num_trajectories / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows), squeeze=False)
    fig.suptitle('Rewards Along Trajectories', fontsize=13)
    fig.canvas.manager.set_window_title("Rewards Along Trajectories")
    
    for i, trajectory in enumerate(trajectories):
        row = i // cols
        col = i % cols
        
        rewards = []
        for state, _ in trajectory:
            state_idx = state_to_index[state]
            reward = state_reward_vector[state_idx]
            rewards.append(reward)

        axes[row, col].plot(rewards)
        axes[row, col].set_xlabel('Step')
        axes[row, col].set_ylabel('Reward')
    
    # Remove any unused subplots
    for i in range(num_trajectories, rows*cols):
        row = i // cols
        col = i % cols
        fig.delaxes(axes[row, col])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to accommodate suptitle
    #plt.show()

def plot_reward_distribution(expert_rewards, random_rewards):
    fig = plt.figure(figsize=(10, 6))
    fig.canvas.manager.set_window_title("TrainVsTest")
    plt.title("Distribution of Cumulative Rewards")
    plt.bar(range(len(expert_rewards)), expert_rewards, alpha=0.5, label='Expert Rewards', color='b')
    plt.bar(range(len(random_rewards)), random_rewards, alpha=0.5, label='Random Rewards', color='r')
    plt.xlabel('Trajectory Index')
    plt.ylabel('Cumulative Reward')
    plt.title('Distribution of Cumulative Rewards')
    plt.legend()
    #plt.show()

def plot_trajectory_and_test_rewards(expert_rewards, test_rewards):
    fig = plt.figure(figsize=(12, 6), label='TrainVsTest')
    
    # Extract indices and rewards
    expert_indices, expert_reward_values = zip(*expert_rewards)
    test_indices, test_reward_values = zip(*test_rewards)
    
    # Plot expert rewards
    plt.bar(expert_indices, expert_reward_values, alpha=0.5, label='Expert Rewards', color='b')
    
    # Plot test rewards
    plt.bar([i + len(expert_rewards) for i in test_indices], test_reward_values, alpha=0.5, label='Test Rewards', color='r')
    
    plt.xlabel('Trajectory Index')
    plt.ylabel('Cumulative Reward')
    plt.title('Distribution of Cumulative Rewards: Expert vs Test')
    plt.legend()
    
    # Adjust x-axis labels
    all_indices = list(expert_indices) + [i + len(expert_rewards) for i in test_indices]
    plt.xticks(all_indices, [str(i) for i in range(len(all_indices))])
    
    #plt.savefig('TrainVsTest')
    plt.tight_layout()
    #plt.show()
