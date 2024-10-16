#!/usr/bin/env python3

import numpy as np
import torch
import features as F
import mdp as MDP
import data_manip as DM
import validation as V
from datetime import datetime
from random_trajectory_generator import generate_random_trajectories
import visualisation as vis
from visualisation import trajectory_reward_visualization
import pickle
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def save_results(reward_functions, theta_list, delta_history, transition_matrices, final_policies, feature_matrix, state_to_index,discretised_demos, trajectories, filename='irl_results.pkl'):
    results = {
        'reward_functions': reward_functions,
        'theta_list': theta_list,
        'delta_history': delta_history,
        'transition_matrices': transition_matrices,
        'final_policies': final_policies,
        'feature_matrix': feature_matrix,
        'state_to_index': state_to_index,
        'discretised_demos': discretised_demos,
        'trajectories': trajectories
    }
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {filename}")

# Preprocess trajectories into segments based on subtasks
def preprocess_trajectories(trajectories):
    grouped_trajectories = {0: [], 1: [], 2: []}
    for trajectory in trajectories.values():
        current_subtask = None
        current_segment = []
        for state, action in trajectory:
            subtask = int(state[-2])  # Assuming subtask is the second-to-last element
            if subtask != current_subtask:
                if current_segment:
                    grouped_trajectories[current_subtask].append(current_segment)
                current_subtask = subtask
                current_segment = []
            current_segment.append((state, action))
        if current_segment:
            grouped_trajectories[current_subtask].append(current_segment)
    return grouped_trajectories
""" grouped_trajectories = {
        0: [trajectory_0_1, trajectory_0_2, ...],
        1: [trajectory_1_1, trajectory_1_2, ...],
        2: [trajectory_2_1, trajectory_2_2, ...]
        }
"""
# compute initial state distribution and terminal states for each subtask
def compute_initial_state_distribution(grouped_trajectories, state_to_index, method='data'):
    p_initial = {0: None, 1: None, 2: None}
    n_states = len(state_to_index)

    for subtask, trajectories in grouped_trajectories.items():
        if method == 'assume':
            p_initial[subtask] = torch.ones(n_states) / n_states
        elif method == 'data':
            initial_state_counts = torch.zeros(n_states)
            for segment in trajectories:
                if segment:
                    initial_state, _ = segment[0]
                    initial_state_index = state_to_index[initial_state]
                    initial_state_counts[initial_state_index] += 1
            p_initial[subtask] = initial_state_counts / initial_state_counts.sum()
        else:
            raise ValueError("Invalid method. Choose 'assume' or 'data'.")

    return p_initial

def compute_terminal_states(grouped_trajectories, state_to_index, method='data'):
    terminal_states = {0: None, 1: None, 2: None}
    n_states = len(state_to_index)

    for subtask, trajectories in grouped_trajectories.items():
        if method == 'assume' or method == 'data':
            terminal_states[subtask] = torch.zeros(n_states, dtype=torch.bool)
            for segment in trajectories:
                if segment:
                    last_state, _ = segment[-1]
                    last_state_index = state_to_index[last_state]
                    terminal_states[subtask][last_state_index] = True
        else:
            raise ValueError("Invalid method. Choose 'assume' or 'data'.")

    return terminal_states

#######################################################################################################################
# expert feature count
def expert_feature_count(grouped_trajectories, feat):
    feature_counts = {0: None, 1: None, 2: None}
    n_features = None

    for subtask, trajectories in grouped_trajectories.items():
        n_states = 0
        if not trajectories:
            continue
        
        if n_features is None:
            n_features = len(feat.feature_vector(trajectories[0][0][0]))
        
        subtask_feature_counts = torch.zeros(n_features).cuda()

        for demo in trajectories:
            n_states += len(demo)
            for state, _ in demo:
                feature_vector = feat.feature_vector(state)
                subtask_feature_counts += torch.tensor(feature_vector).cuda()

        subtask_feature_counts /= n_states
        feature_counts[subtask] = subtask_feature_counts

    return feature_counts, n_features

# D_pi svf from policy
def state_visitation_frequency(trans_matrix, terminal, p_initial, policy, eps = 1e-5):
    
    num_states, _, num_actions = trans_matrix.shape

    # transition_matrix = np.copy(transition_matrix)
    # Use PyTorch tensor instead of NumPy array
    transition_matrix = trans_matrix.clone().detach().to(policy.dtype)
    transition_matrix[terminal, :, :] = 0.0000                                             #!!!!!!!!
    
    # Convert p_initial to the same data type as policy (e.g., float32)
    p_initial = torch.tensor(p_initial, dtype=policy.dtype).cuda()                          #!!!!!!!!

    # Initialize state visitation frequency vector
    D_pi = torch.zeros(num_states, dtype=policy.dtype).cuda() #vector of length num_states

    # set-up transition matrices for each action
    transition_matrix = [transition_matrix[:, :, a] for a in range(num_actions)]
    
    delta = np.inf
    while delta > eps: 
        # Use PyTorch operations for matrix multiplication and element-wise multiplication
        D_pi_new = [transition_matrix[a].t() @ (policy[:, a] * D_pi) for a in range(num_actions)]
    
        # Use PyTorch tensor instead of NumPy array and perform summation on GPU
        D_pi_new = p_initial + torch.stack(D_pi_new).sum(dim=0)
        #D_pi_new = torch.stack(D_pi_new).sum(dim=0)

        # delta, D_pi = np.max(np.abs(D_pi_new - D_pi)), D_pi_new  
        # Use PyTorch operations for maximum and absolute value computation
        delta, D_pi = torch.max(torch.abs(D_pi_new - D_pi)), D_pi_new  
    
    # print("svf calculated")
    return D_pi

# softmax function
def softmax(x1, x2):
    """
    Computes a soft maximum of both arguments.

    In case `x1` and `x2` are arrays, computes the element-wise softmax.

    Args:
        x1: Scalar or ndarray.
        x2: Scalar or ndarray.

    Returns:
        The soft maximum of the given arguments, either scalar or ndarray,
        depending on the input.
    """
    # x_max = np.maximum(x1, x2)
    # x_min = np.minimum(x1, x2)
    # Use PyTorch operations for maximum and minimum computation
    x_max = torch.maximum(x1, x2)
    x_min = torch.minimum(x1, x2)
    return x_max + torch.log(1.0 + torch.exp(x_min - x_max))

def policy_evaluation(transition_matrix, terminal, reward, gamma, max_iterations, eps = 1e-5):
    n_states, _, n_actions = transition_matrix.shape

    reward = reward.to(torch.float32)
    #print(f"Reward stats: min={reward.min().item():.4f}, max={reward.max().item():.4f}, mean={reward.mean().item():.4f}")

    reward_terminal = torch.full((n_states,), -1e20, dtype=torch.float32, device='cuda')
    reward_terminal[terminal] = 0.0                #!!!!!!!!

    p = [transition_matrix[:, :, a].clone().detach().to(torch.float32) for a in range(n_actions)]

    v = torch.full((n_states,), -1e20, dtype=torch.float32, device='cuda')

    delta = np.inf
    iterations = 0

    while delta > eps and iterations < max_iterations:
        v_old = v

        q = torch.stack([reward + gamma * torch.logsumexp(torch.log(p[a] + 1e-10) + v_old.unsqueeze(1), dim=1) for a in range(n_actions)], dim=1)

        # print(f"Iteration {iterations}:")
        # print(f"  Q stats: min={q.min().item():.4f}, max={q.max().item():.4f}, mean={q.mean().item():.4f}")

        v = reward_terminal
        for a in range(n_actions):
            v = softmax(v, q[:, a])

        #print(f"  V stats: min={v.min().item():.4f}, max={v.max().item():.4f}, mean={v.mean().item():.4f}")

        delta = torch.max(torch.abs(torch.exp(v) - torch.exp(v_old)))
        #delta = torch.max(torch.abs(v - v_old))

        if delta < eps:
            break
        iterations += 1

        if torch.isnan(v).any() or torch.isinf(v).any():
            print("NaN or Inf values encountered in policy evaluation. Stopping.")
            break

    if iterations == max_iterations:
        print("Maximum iterations reached in policy evaluation.")

    policy = torch.exp(q - v.unsqueeze(1))
    #print(f"Policy stats: min={policy.min().item():.4f}, max={policy.max().item():.4f}, mean={policy.mean().item():.4f}")
    
    return policy

# expected state visitation frequency from policy and theta
def compute_expected_svf(transition_matrix, terminal, p_initial, reward, gamma, max_policy_iterations):
    
    policy = policy_evaluation(transition_matrix, terminal, reward, gamma, max_policy_iterations) 
    # Ensure policy is valid (sum to 1 for each state)
    policy = policy / (policy.sum(dim=1, keepdim=True) + 1e-10)
    
    D_pi = state_visitation_frequency(transition_matrix, terminal, p_initial, policy)
    
    return D_pi, policy

# Maximum Causal Entropy Inverse Reinforcement Learning
def mce_irl(transition_matrix, state_to_index, features, feature_matrix, trajectories, gamma, max_policy_iterations, eps=1e-4, init_term_method='data'):
    grouped_trajectories = preprocess_trajectories(trajectories)
    e_features, n_features = expert_feature_count(grouped_trajectories, features)
    p_initial = compute_initial_state_distribution(grouped_trajectories, state_to_index, method=init_term_method)
    terminal_states = compute_terminal_states(grouped_trajectories, state_to_index, method=init_term_method)

    n_subtasks, n_states, _, n_actions = transition_matrix.shape
    
    reward_functions = []
    theta_list = []
    delta_history = []
    policies = []

    for subtask in range(3):
        if e_features[subtask] is None:
            print(f"No data for subtask {subtask}, skipping.")
            reward_functions.append(None)
            theta_list.append(None)
            continue

        print(f"Processing subtask {subtask}")
        
        # Initialize theta with random values and set delta to infinity  
        theta = torch.rand(n_features, device='cuda') * 0.0001           #!!! could be random values
        delta = np.inf

        # Hyperparameters for the learning rate decay  
        lr0 = 0.1
        decay_rate = 1.0
        decay_steps = 1
        k = 0
        
        # monitor reward function stability and apply early stopping
        reward_history = []
        patience = 10
        best_delta = float('inf')
        no_improvement_count = 0

        while delta > eps:               
            if delta < best_delta:                   # early stopping criterion
                best_delta = delta
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"Early stopping at iteration {k}")
                break

            theta_old = theta.clone()
            reward = feature_matrix.cuda() @ theta    # NsxNf * Nfx1 = Nsx1, vector of length num_states, each index corresponds to the reward of the state

            # check reward function stability
            reward_history.append(reward.mean().item())
            if len(reward_history) > 10:
                reward_stability = np.std(reward_history[-10:])
                print(f"Reward stability: {reward_stability:.6f}")
                if reward_stability < 1e-6:
                    print(f"Reward function stabilized at iteration {k}")
                    break

            # compute expected state visitation frequency
            e_svf, policy = compute_expected_svf(transition_matrices[subtask], terminal_states[subtask], p_initial[subtask], reward, gamma, max_policy_iterations)
            grad = e_features[subtask].cuda() - feature_matrix.t().cuda() @ e_svf

            if torch.abs(grad).max() > 1e5:
                print("Extreme gradient values encountered. Clipping gradients.")
                grad = torch.clamp(grad, min=-1e5, max=1e5)

            lr = lr0 / (1.0 + decay_rate * (k // decay_steps))  # decay formula for learning rate
            k += 1

            epsilon = 1e-8
            theta_new = theta * torch.exp(lr * grad)
            # Update theta using a more stable update rule
            # theta_new = torch.clamp(theta_new, min=epsilon)  # Ensure all values are positive
            theta = theta_new / theta_new.sum()

            delta = torch.max(torch.abs(theta_old - theta))  # takes max of the absolute difference between the old and new theta
            delta_history.append(delta.item())  # Append delta to history

            print(f"Iteration {k}:")
            print(f"  Delta: {delta.item():.4f}")

            if torch.isnan(theta).any() or torch.isnan(delta):
                print("NaN values encountered. Stopping optimization.")
                break

            if k >= 1000:
                print("Maximum iterations reached. Stopping optimization.")
                break

        print(f"Iterations for Theta convergence {k}:")
        print(f"  Max grad: {torch.max(grad).item():.4f}, Min grad: {torch.min(grad).item():.4f}")
        print(f"  Max theta: {torch.max(theta).item():.4f}, Min theta: {torch.min(theta).item():.4f}")
        print(f"  Delta: {delta.item():.4f}")

        reward_function = (feature_matrix.cuda() @ theta).cpu()
        reward_functions.append(reward_function)
        theta_list.append(theta)
        policies.append(policy)

    return reward_functions, theta_list, delta_history, policies

#####################################################################################################################################################################################
def compute_cumulative_rewards(trajectories, theta_list, feat):
    """
    Compute cumulative rewards for each trajectory and the average across all trajectories.

    Args:
    trajectories (dict): Dictionary of trajectories, where keys are filenames and values are lists of (state, action) tuples.
    theta (torch.Tensor): The learned reward function parameters.
    feat (Features): The feature object used to compute state features.

    Returns:
    dict: A dictionary containing cumulative rewards for each trajectory and the average reward.
    """
    cumulative_rewards = {}
    total_reward = 0

    for filename, trajectory in trajectories.items():
        trajectory_reward = 0
        for state, _ in trajectory:
            subtask = int(state[-2])  # Assuming subtask is the second-to-last element
            feature_vector = torch.tensor(feat.feature_vector(state), dtype=torch.float32).cuda()
            theta = theta_list[subtask] * (1+subtask*0.1)
            state_reward = torch.dot(feature_vector, theta).item()
            trajectory_reward += state_reward
        
        trajectory_reward /= len(trajectory) 
        cumulative_rewards[filename] = trajectory_reward*1000
        total_reward += trajectory_reward

    average_reward = total_reward / len(trajectories)
    

    return cumulative_rewards, average_reward

def find_min_max_rewards(cumulative_rewards):
    """
    Find the minimum and maximum cumulative rewards and their corresponding filenames.

    Args:
    cumulative_rewards (dict): Dictionary containing cumulative rewards for each trajectory and the average reward.

    Returns:
    tuple: A tuple containing (min_filename, min_reward, max_filename, max_reward).
    """
    min_reward = float('inf')
    max_reward = float('-inf')
    min_filename = None
    max_filename = None

    for filename, reward in cumulative_rewards.items():
        if filename == 'average':
            continue
        if reward < min_reward:
            min_reward = reward
            min_filename = filename
        if reward > max_reward:
            max_reward = reward
            max_filename = filename

    return min_filename, min_reward, max_filename, max_reward

def plot_theta_values(theta_list):
    for i, theta in enumerate(theta_list):
        if theta is not None:
            fig = go.Figure()
            theta_np = theta.cpu().numpy()
            fig.add_trace(
                go.Bar(x=list(range(len(theta_np))), y=theta_np, name=f'Subtask {i}')
            )
            fig.update_layout(
                title=f"Theta Values for Subtask {i}",
                xaxis_title="Feature Index",
                yaxis_title="Theta Value",
                height=600,
                width=1200
            )
            fig.show()

def plot_trajectory_rewards(trajectories, theta_list, feat):
    n_trajectories = len(trajectories)
    n_cols = 4
    n_rows = (n_trajectories + n_cols - 1) // n_cols

    fig = make_subplots(rows=n_rows, cols=n_cols, 
                        subplot_titles=[f'Trajectory {filename}' for filename in trajectories.keys()],
                        horizontal_spacing= 0.1 ,vertical_spacing=0.2)

    for idx, (filename, trajectory) in enumerate(trajectories.items()):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        rewards = []
        subtask_changes = []
        current_subtask = int(trajectory[0][0][-2])

        for i, (state, _) in enumerate(trajectory):
            subtask = int(state[-2])
            if subtask != current_subtask:
                subtask_changes.append(i)
                current_subtask = subtask

            feature_vector = torch.tensor(feat.feature_vector(state), dtype=torch.float32).cuda()
            theta = theta_list[subtask]
            state_reward = torch.dot(feature_vector, theta).item()
            rewards.append(state_reward)

        fig.add_trace(
            go.Scatter(x=list(range(len(rewards))), y=rewards, mode='lines', name=f'Trajectory {idx+1}'),
            row=row, col=col
        )

        for change in subtask_changes:
            fig.add_shape(
                type="line", x0=change, y0=min(rewards), x1=change, y1=max(rewards),
                line=dict(color="Red", width=2, dash="dash"),
                row=row, col=col
            )

        fig.update_xaxes(title_text="Time Step", row=row, col=col)
        fig.update_yaxes(title_text="State Reward", row=row, col=col)

    fig.update_layout(height=300*n_rows, width=1200, 
                      title_text="State Rewards Along Trajectories",
                      showlegend=False)
    fig.show()


if __name__ == '__main__':
    n_p_bins = 200
    n_v_bins = 200
    n_f_bins = 200
    extract_data = False
    if extract_data:
        DM.extract_and_save_filtered_data('train10.07', 'train_extracted', 100)
        DM.extract_and_save_filtered_data('fails', 'fails_extracted', 100)
        DM.extract_and_save_filtered_data('test10.07', 'test_extracted', 100)

        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Data Extraction and Filtering Completed Last on {current_datetime}")
        print("------------------------------------------")

  


    if os.path.exists('irl_results.pkl'):
        # Load saved results
        with open('irl_results.pkl', 'rb') as f:
            results = pickle.load(f)
        reward_functions = results['reward_functions']
        theta_list = results['theta_list']
        delta_history = results['delta_history']
        transition_matrices = results['transition_matrices']
        final_policies = results['final_policies']
        feature_matrix = results['feature_matrix']
        state_to_index = results['state_to_index']
        discretised_demos = results['discretised_demos']
        train_trajectories = results['trajectories']
        feat = F.Features(train_trajectories, 20) 
        print("Loaded saved results.")
    else:  
        print("Running IRL...")
        # Load the data, discretise it and create trajectories
        discretised_demos = DM.data_manip(n_p_bins, n_v_bins, n_f_bins, 'train_extracted', 'train_discretised')[0]
        train_trajectories =  MDP.create_trajectories(discretised_demos, 'train_trajectories')   
        feat = F.Features(train_trajectories, n_bins=20) 
        # Run IRL
        transition_matrices, _, _ = MDP.transition_probability_matrix(train_trajectories)
        transition_matrices = transition_matrices.cuda()

        # create a state mapper object
        state_mapper = F.StateMappingStrategies(feat)
        feature_matrix = state_mapper.get_feature_matrix('original')

        state_to_index = feat.state_to_index
        reward_functions, theta_list, delta_history, final_policies = mce_irl(transition_matrices, state_to_index, feat, feature_matrix, train_trajectories, gamma=0.99, max_policy_iterations=100000, init_term_method='data')
        save_results(reward_functions, theta_list, delta_history, transition_matrices, final_policies, feature_matrix, state_to_index, discretised_demos, train_trajectories)
        print("IRL completed, results saved.")
        for subtask, (reward_function, theta) in enumerate(zip(reward_functions, theta_list)):
            if reward_function is not None:
                print(f"Subtask {subtask}:")
                print(f"Reward function shape: {reward_function.shape}")
                print(f"Theta shape: {theta.shape}")
                print(f"theta: {theta}")


    

    # Test cumulative rewards
    test_disc_demos = DM.data_manip(n_p_bins, n_v_bins, n_f_bins, 'test_extracted', 'test_discretised')[0]
    test_trajectories = MDP.create_trajectories(test_disc_demos, 'test_trajectories')
    test_cumulative_rewards, test_average = compute_cumulative_rewards(test_trajectories, theta_list, feat)
    for filename, reward in test_cumulative_rewards.items():
        print(f"{filename}: {reward}")
    min_filename, min_reward, max_filename, max_reward = find_min_max_rewards(test_cumulative_rewards)
    

    # Fails cumulative rewards 
    fails_discretised_demos = DM.data_manip(n_p_bins, n_v_bins, n_f_bins, 'fails_extracted', 'fails_discretised')[0]
    fails_trajectories =  MDP.create_trajectories(fails_discretised_demos, 'fails_trajectories')  
    fails_cumulative_rewards, fails_average = compute_cumulative_rewards(fails_trajectories, theta_list, feat)
    for filename, reward in fails_cumulative_rewards.items():
        print(f"{filename}: {reward}")
    min_filename, min_reward, max_filename, max_reward = find_min_max_rewards(fails_cumulative_rewards)
    
    print(f"test minimum reward: {min_reward} (Filename: {min_filename})")
    print(f"test maximum reward: {max_reward} (Filename: {max_filename})")
    print('----------------------------------------------------------------------')
    print(f"fails minimum reward: {min_reward} (Filename: {min_filename})")
    print(f"fails maximum reward: {max_reward} (Filename: {max_filename})")
    print('----------------------------------------------------------------------')
    print(f"test average reward: {test_average}")
    print(f"fails average reward: {fails_average}")

    # Plotting
    MDP.plot_trajectory_features(train_trajectories, 'train_trajectory_plots')
    MDP.plot_trajectory_features(test_trajectories, 'test_trajectories_plots')
    MDP.plot_trajectory_features(fails_trajectories, 'fails_trajectories_plots')
   
    # plot_theta_values(theta_list)
    # plot_trajectory_rewards(trajectories, theta_list, feat)
    # plot_trajectory_rewards(fails_trajectories, theta_list, feat)
    # plot_trajectory_rewards(test_trajectories, theta_list, feat)

    # Validation
    roc_auc = V.perform_roc_analysis([r for r in test_cumulative_rewards.values() if isinstance(r, (int, float))],
                               [r for r in fails_cumulative_rewards.values() if isinstance(r, (int, float))])
    print(f"ROC AUC: {roc_auc}")

    # feature importance
    feature_names = ['Avg F/V Ratio', 'Force Trend', 'Velocity Smoothness', 'Position',
                    'Lift Acceleration', 'Force Impulse', 'Velocity Reversal Freq',
                    'Subtask Transition Prob', 'Time in Subtask', 'Distance to Centroid',
                    'Subtask', 'Timestep']
    V.analyze_feature_importance(theta_list, feature_names,n_features=12,n_bins=20, n_ohe_features=10)

    # Statistical test for significant difference between successfull and unsuccsessfull trajectories
    V.analyze_reward_differentiation(test_cumulative_rewards, fails_cumulative_rewards)

    # Visualisation of theta
    V.plot_reward_distributions(test_cumulative_rewards, fails_cumulative_rewards)

    # Classification metrics
    V.calculate_classification_metrics(test_cumulative_rewards, fails_cumulative_rewards)