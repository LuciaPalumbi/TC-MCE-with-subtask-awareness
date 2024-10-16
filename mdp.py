#!/usr/bin/env python3
import os
import json
import numpy as np
import data_manip as DM
import features as F
import itertools
import torch
from scipy import sparse
from collections import deque
import matplotlib.pyplot as plt
from data_manip import reset

from collections import deque

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def create_trajectories(discretised_demos, trajectories_folder):
    reset(trajectories_folder)
    trajectories = {}
    for filename, demo in discretised_demos.items() :
        # separate filename and extension
        filename = filename.split('.')[0]
        trajectory = []
        window = deque(maxlen=11)  # Current state + up to 10 past states
        """ new deque object with a maximum length of 11. 
            A deque, short for "double-ended queue," is a data structure from
            Python's collections module that allows for efficient appending and 
            popping of elements from both ends."""
        
        for i in range(len(demo)):
            current_state = demo[i]
            window.append(current_state)
            
            # Start computing features as soon as we have at least 2 states
            if len(window) >= 2:
                p_current, v_current, f_current, lift_current, subtask_current = current_state
                timestep = i*9e-3 #seconds
                # Calculate features based on available window
                positions = np.array([state[0] for state in window])
                velocities = np.array([state[1] for state in window])
                forces = np.array([state[2] for state in window])
                lifts = np.array([state[3] for state in window])
                subtasks = np.array([state[4] for state in window])
                time_indices = np.arange(len(window))
                total_time = (len(window) - 1) * 0.009  # in seconds
                # Average Force/Velocity Ratio:
                avg_fv_ratio = np.mean([abs(state[2]) / abs(state[1]) if np.abs(state[1]) > 1e-5 else abs(state[2]) / 1e-5 for state in window])
                # Force Trend, slope of force over time:
                force_trend, _ = np.polyfit(time_indices, forces, 1)/9e-3
                # Velocity Smoothness, average jerk:
                if len(velocities) >= 3:
                    jerk = np.diff(velocities, n=2)
                    velocity_smoothness = np.mean(np.abs(jerk))
                else:
                    velocity_smoothness = 0
                # Position Change Rate, slope of position over time:
                position_change_rate = (p_current - positions[0]) /total_time if total_time > 0 else 0
                # Lift Acceleration, slope of lift over time:
                if len(lifts) >= 3:
                    # Calculate velocity (first derivative)
                    d_velocity = np.diff(lifts) / 0.009  # 9ms = 0.009s between states
                    # Calculate acceleration (second derivative)
                    acceleration = np.diff(d_velocity) / 0.009
                    # Return the average acceleration
                    lift_acceleration = np.mean(acceleration)
                # Return 0 if we don't have enough data point
                else:
                    lift_acceleration = 0
                # Force Impulse, area under the force curve:
                force_impulse = impulse = np.sum(forces[1:]) * 0.009
                # Velocity Reversal Frequency, which is the number of times the velocity changes sign:
                velocity_reversal_freq = np.mean(np.diff(np.sign(velocities)) != 0)
                # Subtask Transition Probability, which is the probability of changing subtasks:
                subtask_transition_prob = np.sum(np.diff(subtasks) != 0) / (len(subtasks) - 1)

                # Time in Subtask, which is the time spent in the current subtask:
                subtask_changes = np.where(np.diff(subtasks) != 0)[0]
                if len(subtask_changes) > 0:
                    # Time since last subtask change
                    time_in_subtask = (len(subtasks) - 1 - subtask_changes[-1]) * 0.009  # in seconds
                else:
                    # If no changes, time is the entire window
                    time_in_subtask = (len(subtasks) - 1) * 0.009  # in seconds
                
                # Distance to Subtask
                current_subtask = window[-1][4]
                subtask_states = np.array([state[:4] for state in window if state[4] == current_subtask])
                centroid = np.mean(subtask_states, axis=0)
                distance_to_centroid =  np.linalg.norm(window[-1][:4] - centroid)
                
                # Create new state with features
                new_state = (avg_fv_ratio, 
                             force_trend,
                             velocity_smoothness, 
                             p_current, 
                             lift_acceleration, 
                             force_impulse, 
                             velocity_reversal_freq,
                             subtask_transition_prob, 
                             time_in_subtask, 
                             distance_to_centroid,
                             subtask_current, 
                             timestep
                             )
                # Calculate action (change in position)
                action = round(p_current - window[-2][0], 5)  # Compare with previous state
                
                # if the new state is the same as the previous state, skip it
                old_state = trajectory[-1][0] if trajectory else None
                if new_state == old_state:
                    continue
                else:
                    trajectory.append((new_state, action))
        
        trajectories[filename] = trajectory
    
    # Save trajectories to files
    if not os.path.exists(trajectories_folder):
        os.makedirs(trajectories_folder)

    for filename, trajectory in trajectories.items():
        file_path = f'{trajectories_folder}/trajectory_{filename}.json'
        with open(file_path, 'w') as file:
            json.dump(trajectory, file, indent=4, cls=NumpyEncoder)

    return trajectories

def analyze_trajectories(trajectories):
    total_states = 0
    total_non_zero_actions = 0
    total_identical_states = 0

    for filename, trajectory in trajectories.items():
        non_zero_actions = 0
        identical_states = 0
        
        for i in range(len(trajectory)):
            state, action = trajectory[i]
            
            # Count non-zero actions
            if action != 0.0:
                non_zero_actions += 1
            
            # Count identical adjacent states
            if i > 0:
                prev_state, _ = trajectory[i-1]
                if state == prev_state:
                    identical_states += 1
        
        total_states += len(trajectory)
        total_non_zero_actions += non_zero_actions
        total_identical_states += identical_states
        
        print(f"\nTrajectory: {filename}")
        print(f"Total states: {len(trajectory)}")
        print(f"Non-zero actions: {non_zero_actions} ({non_zero_actions/len(trajectory)*100:.2f}%)")
        print(f"Identical adjacent states: {identical_states} ({identical_states/(len(trajectory)-1)*100:.2f}%)")

    print("\nOverall Statistics:")
    print(f"Total states across all trajectories: {total_states}")
    print(f"Total non-zero actions: {total_non_zero_actions} ({total_non_zero_actions/total_states*100:.2f}%)")
    print(f"Total identical adjacent states: {total_identical_states} ({total_identical_states/(total_states-len(trajectories))*100:.2f}%)")

def generate_all_states(positions, velocities, forces, lifts):
    """
    Generate all possible combinations of positions, velocities, forces, and lifts.
    
    Args:
    positions (list): List of possible position values.
    velocities (list): List of possible velocity values.
    forces (list): List of possible force values.
    lifts (list): List of possible lift values.
    
    Returns:
    list: A list of tuples, where each tuple represents a possible state (position, velocity, force, lift).
    """
    # Use itertools.product to generate all combinations
    all_states = list(itertools.product(positions, velocities, forces, lifts))
    
    return all_states

def transition_probability_matrix(trajectories, num_subtasks=3):
    # Count the number of unique states and actions
    states = set()
    actions = set()
    for _, trajectory in trajectories.items():
        for state, action in trajectory:
            states.add(state)
            actions.add(action)
    
    num_states = len(states)
    num_actions = len(actions)

    actions = sorted(actions)
    states = sorted(states, key=lambda x: x[0])

    print("-----------------------------------------------------------")
    print("From Transition Probability Matrix:")
    print(f"Number of states: {num_states}")
    print(f"Number of actions: {num_actions}")
    print(f"Actions: {actions}")
    print("-----------------------------------------------------------")

    # Create a mapping from states and actions to indices
    state_to_index = {state: index for index, state in enumerate(states)}
    action_to_index = {action: index for index, action in enumerate(actions)}

    # Initialize the transition probability matrices using sparse matrices
    transition_matrices = [
        [sparse.lil_matrix((num_states, num_states)) for _ in range(num_actions)]
        for _ in range(num_subtasks)
    ]
    
    # Count the occurrences of transitions for each action
    for filename, trajectory in trajectories.items():
        for i in range(len(trajectory) - 1):
            current_state, action = trajectory[i]
            next_state, _ = trajectory[i + 1]
            subtask = int(current_state[-2])  # Assuming subtask is the second-to-last element
            
            current_state_index = state_to_index[current_state]
            next_state_index = state_to_index[next_state]
            if action in action_to_index:
                action_index = action_to_index[action]
                transition_matrices[subtask][action_index][current_state_index, next_state_index] += 1
            else:
                print(f"Warning: Action {action} not in action set, skipping.")
    
    # Normalize each transition matrix
    for subtask in range(num_subtasks):
        for action_index in range(num_actions):
            row_sums = transition_matrices[subtask][action_index].sum(axis=1).A.flatten()
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            transition_matrices[subtask][action_index] = transition_matrices[subtask][action_index].multiply(1 / row_sums[:, np.newaxis])
    
    # Convert to PyTorch tensors
    torch_transition_matrices = torch.zeros((num_subtasks, num_states, num_states, num_actions), dtype=torch.float32)
    for subtask in range(num_subtasks):
        for action in range(num_actions):
            torch_transition_matrices[subtask, :, :, action] = torch.from_numpy(transition_matrices[subtask][action].toarray().astype(np.float32))

    return torch_transition_matrices, state_to_index, action_to_index

# def transition_probability_matrix(trajectories, num_subtasks = 3, fixed_actions=None):
#     # Count the number of unique states and actions
#     states = set()
#     actions = set()
#     for _, trajectory in trajectories.items():
#         for state, action in trajectory:
#             states.add(state)
#             actions.add(action)
    
#     num_states = len(states)
#     num_actions = len(actions)

#     actions = sorted(actions)
#     #order states by position value
#     states = sorted(states, key=lambda x: x[0])

#     print("-----------------------------------------------------------")
#     print("From Transition Probability Matrix:")
#     print(f"Number of states: {num_states}")
#     print(f"Number of actions: {num_actions}")
#     # print the action set
#     print(f"Actions: {actions}")
#     print("-----------------------------------------------------------")

#     # Create a mapping from states and actions to indices
#     state_to_index = {state: index for index, state in enumerate(states)}
#     action_to_index = {action: index for index, action in enumerate(actions)}

#     # print(f"len(state_to_index): {len(state_to_index)}") 

#     # Initialize the transition probability matrix
#     transition_matrices = [np.zeros((num_states, num_states, num_actions)) for _ in range(num_subtasks)]
    
#     for filename, trajectory in trajectories.items():
#         for i in range(len(trajectory) - 1):
#             current_state, action = trajectory[i]
#             next_state, _ = trajectory[i + 1]
#             subtask = current_state[-2]  # Assuming subtask is the second-to-last element
            
#             current_state_index = state_to_index[current_state]
#             next_state_index = state_to_index[next_state]
#             action_index = action_to_index[action]
#             if action in action_to_index:
#                 action_index = action_to_index[action]
#                 transition_matrices[subtask][current_state_index, next_state_index, action_index] += 1
#             else:
#                 print(f"Warning: Action {action} not in action set, skipping.")
            
    
#     # Normalize each transition matrix
#     for subtask in range(num_subtasks):
#         for state_index in range(num_states):
#             for action_index in range(num_actions):
#                 total = np.sum(transition_matrices[subtask][state_index, :, action_index])
#                 if total > 0:
#                     transition_matrices[subtask][state_index, :, action_index] /= total
    
#     return transition_matrices, state_to_index, action_to_index

def mdp(discretised_demos, trajectories_folder):
    
    # get the trajectories from the discretised demonstrations
    trajectories = create_trajectories(discretised_demos, trajectories_folder)
    # get the transition matrix, state to index and action to index mappings
    transition_matrix, state_to_index, action_to_index = transition_probability_matrix(trajectories)

    return trajectories, transition_matrix, state_to_index, action_to_index

def visualize_trajectory(trajectories, trajectory_name):
    if trajectory_name not in trajectories:
        print(f"Trajectory '{trajectory_name}' not found.")
        return
    
    trajectory = trajectories[trajectory_name]
    
    # Extract features and timesteps
    features = np.array([state for state, _ in trajectory])
    timesteps = features[:, -1]  # Assuming timestep is the last feature
    
    feature_names = [
        'Avg F/V Ratio', 'Force Trend', 'Velocity Smoothness', 'Position Change Rate',
        'Lift Acceleration', 'Force Impulse', 'Velocity Reversal Freq',
        'Subtask Transition Prob', 'Time in Subtask', 'Distance to Centroid',
        'Subtask', 'Timestep'
    ]
    
    # Group related features
    feature_groups = [
        ('Force and Velocity', [0, 1, 2, 6]),  # F/V Ratio, Force Trend, Velocity Smoothness, Velocity Reversal Freq
        ('Position and Lift', [3, 4, 5]),  # Position Change Rate, Lift Acceleration, Force Impulse
        ('Subtask Information', [7, 8, 9, 10])  # Subtask Transition Prob, Time in Subtask, Distance to Centroid, Subtask
    ]
    
    fig, axes = plt.subplots(len(feature_groups), 1, figsize=(15, 5*len(feature_groups)), sharex=True)
    fig.suptitle(f'Trajectory Analysis: {trajectory_name}', fontsize=16)
    
    for i, (group_name, feature_indices) in enumerate(feature_groups):
        ax = axes[i]
        ax.set_title(group_name)
        
        for j in feature_indices:
            ax.plot(timesteps, features[:, j], label=feature_names[j])
        
        ax.legend()
        ax.grid(True)
        
        if i == len(feature_groups) - 1:
            ax.set_xlabel('Time (s)')
    
    # Plot actions
    actions = [action for _, action in trajectory]
    ax_actions = axes[-1].twinx()
    ax_actions.plot(timesteps, actions, 'r-', alpha=0.5, label='Action')
    ax_actions.set_ylabel('Action', color='r')
    ax_actions.tick_params(axis='y', labelcolor='r')
    ax_actions.legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()

def plot_trajectory_features(trajectories, output_folder='trajectory_plots'):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Determine the number of features
    num_features = len(next(iter(trajectories.values()))[0][0]) - 1  # Subtract 1 to exclude the timestep

    # Feature names (update these if they're different)
    feature_names = [
        'Avg F/V Ratio', 'Force Trend', 'Velocity Smoothness', 'Curent Position',
        'Lift Acceleration', 'Force Impulse', 'Velocity Reversal Freq',
        'Subtask Transition Prob', 'Time in Subtask', 'Distance to Centroid',
        'Subtask'
    ]

    # Create a plot for each feature
    for feature_idx in range(num_features):
        plt.figure(figsize=(12, 6))
        
        for filename, trajectory in trajectories.items():
            # Extract feature values and timesteps
            feature_values = [state[feature_idx] for state, _ in trajectory]
            timesteps = [state[-1] for state, _ in trajectory]
            
            # Plot this trajectory
            plt.plot(timesteps, feature_values, label=filename, alpha=0.7)
        
        plt.title(f'{feature_names[feature_idx]} vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel(feature_names[feature_idx])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(output_folder, f'{feature_names[feature_idx].replace("/", "_")}_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Plots have been saved in the '{output_folder}' directory.")


if __name__ == '__main__':
    n_p_bins = 200
    n_v_bins = 200
    n_f_bins = 200
    discretised_test = DM.data_manip(n_p_bins, n_v_bins, n_f_bins, 'test_extracted', 'test_discretised')[0]
    discretised_demos = DM.data_manip(n_p_bins, n_v_bins, n_f_bins, 'train_extracted', 'train_discretised')[0]
    discretised_fails = DM.data_manip(n_p_bins, n_v_bins, n_f_bins, 'fails_extracted', 'fails_discretised')[0]
    train_trajectories =  create_trajectories(discretised_demos, 'train_trajectories')  
    test_trajectories = create_trajectories(discretised_test, 'test_trajectories')
    fails_trajectories = create_trajectories(discretised_fails, 'fails_trajectories')

    plot_trajectory_features(train_trajectories, 'train_trajectory_plots')
    plot_trajectory_features(test_trajectories, 'test_trajectories_plots')
    plot_trajectory_features(fails_trajectories, 'fails_trajectories_plots')



   



    