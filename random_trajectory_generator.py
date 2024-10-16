#!/usr/bin/env python3

import random

import random

def generate_random_trajectories(positions, velocities, forces, lifts, actions, num_trajectories=16, max_trajectory_length=70):
    trajectories = []
    actions_list = list(actions)  # Convert dict_keys to a list
    
    for _ in range(num_trajectories):
        trajectory = []
        current_state = (
            random.choice(positions),
            random.choice(velocities),
            random.choice(forces),
            random.choice(lifts)
        )
        
        for _ in range(max_trajectory_length):
            action = random.choice(actions_list)
            trajectory.append((current_state, action))
            
            # Generate next state
            current_position_index = positions.index(current_state[0])
            next_position_index = current_position_index + positions.index(min(positions, key=lambda x: abs(x - (current_state[0] + action))))
            next_position_index = max(0, min(next_position_index, len(positions) - 1))
            next_position = positions[next_position_index]
            
            next_velocity = random.choice(velocities)
            next_force = random.choice(forces)
            next_lift = random.choice(lifts)
            
            next_state = (next_position, next_velocity, next_force, next_lift)
            
            # Check if the trajectory should end (e.g., reaching a terminal state)
            if next_lift > 3.000 and next_velocity <= 0.02:
                trajectory.append((next_state, 0))  # Add terminal state with action 0
                break
            
            current_state = next_state
        
        trajectories.append(trajectory)
    
    return trajectories

# To Copy in the main file:
"""
   #Compare with random trajectories
     #Compare with random trajectories
    E = 0
    R = 0
    random_trajectory_rewards = []
    for i in range(100):
        # Generate random trajectories
        random_trajectories = generate_random_trajectories(positions, velocities, forces, lifts, list(action_to_index.keys()))
        
        # Compute cumulative rewards
        random_avg_reward, random_rewards = compute_average_cumulative_reward(random_trajectories, Theta, feat)
        random_trajectory_rewards.extend(random_rewards)
        
        if random_avg_reward > expert_avg_reward:
            R += 1
        else:
            E += 1

    print(f"Expert: {E}, Random: {R}")

    # Plot reward distributions
    plot_reward_distribution(expert_trajectory_rewards, random_trajectory_rewards[:len(expert_trajectory_rewards)])

    # Visualize random trajectory rewards
    #plot_trajectory_rewards(random_trajectories, random_state_to_index, random_feature_matrix, Theta)
"""