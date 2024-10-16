#!/usr/bin/env python3

import numpy as np
import data_manip as DM
import mdp as MDP
from scipy.spatial import cKDTree
import torch

class Features:
    def __init__(self, trajectories, n_bins):
        self.feature_ranges = self.compute_feature_ranges(trajectories)
        self.state_to_index = self.create_state_index_mapping(trajectories)
        self.n_bins = n_bins
        self.num_features = self.get_num_features()
        print(f"n_bins set to: {self.n_bins}")  # Add this line
        self.num_features = self.get_num_features()
        print(f"num_features set to: {self.num_features}")
    
    def compute_feature_ranges(self, trajectories):
        feature_names = ['avg_fv_ratio', 'force_trend', 'velocity_smoothness', 
                         'position', 'lift_acceleration', 'force_impulse', 
                         'velocity_reversal_freq', 'subtask_transition_prob', 
                         'time_in_subtask', 'distance_to_centroid', 'subtask']    # 'timestep' removed
        
        feature_values = {name: [] for name in feature_names}
        
        for trajectory in trajectories.values():
            for state, _ in trajectory:  
                # remove last object in state list
                state = state[:-1]
                for i, name in enumerate(feature_names):
                    feature_values[name].append(state[i])
        
        feature_ranges = {}
        for name in feature_names:
            if name == 'subtask':
                feature_ranges[name] = (min(feature_values[name]), max(feature_values[name]))
            else:
                feature_ranges[name] = (np.min(feature_values[name]), np.max(feature_values[name]))
        
        return feature_ranges
    
    def feature_vector(self, state):
        """
        Create a feature vector for a given state.
        
        Args:
        state (tuple): (avg_fv_ratio, force_trend, velocity_smoothness, position, 
                        lift_acceleration, force_impulse, velocity_reversal_freq,
                        subtask_transition_prob, time_in_subtask, distance_to_centroid,
                        subtask, timestep)
        feature_ranges (dict): Dictionary containing the range for each feature
        
        Returns:
        np.array: Feature vector
        """
        feature_vector = []
        
        # Create bins for continuous feat
        continuous_features = ['avg_fv_ratio', 'force_trend', 'velocity_smoothness', 
                            'position', 'lift_acceleration', 'force_impulse', 
                            'velocity_reversal_freq', 'subtask_transition_prob', 
                            'time_in_subtask', 'distance_to_centroid']
        
        for i, feature in enumerate(continuous_features):
            feature_min, feature_max = self.feature_ranges[feature]
            bin_index = int(np.clip((state[i] - feature_min) / (feature_max - feature_min) * self.n_bins, 0, self.n_bins - 1))
            one_hot = [0] * self.n_bins
            one_hot[bin_index] = 1
            feature_vector.extend(one_hot)
        
        subtask_fail = [0]
        if state[3] == 1:
            subtask_fail = [-1]
        feature_vector.extend(subtask_fail)
       
        # One-hot encoding for subtask
        subtask_one_hot = [0] * 3  # Assuming 3 subtasks
        subtask_one_hot[int(state[-2])] = state[-2]       
        feature_vector.extend(subtask_one_hot)
        
        # # Normalize timestep
        # normalized_timestep = state[-1] / self.feature_ranges['timestep'][1]  # Assuming min timestep is 0
        # feature_vector.append(normalized_timestep)

        # negative feature for closing the gripper
       
        
        return np.array(feature_vector)
    
    def get_num_features(self):
        n_continuous_features = 10
        n_subtasks = 3
        return self.n_bins * n_continuous_features + n_subtasks + 1 # + 1 for timestep

    def create_state_index_mapping(self, trajectories):
        unique_states = set()
        for trajectory in trajectories.values():
            unique_states.update(state for state, _ in trajectory)
        return {state: idx for idx, state in enumerate(unique_states)}
    
###################################################################################################################################################################
class StateMappingStrategies:
    def __init__(self, feat):
        self.feat = feat

    def get_feature_matrix(self, method):
        if method == 'original':
            return self.original_feature_matrix()
        else:
            raise ValueError(f"Unknown method: {method}")

    def original_feature_matrix(self):
        num_states = len(self.feat.state_to_index)
        print(f"Number of states: {num_states}")
        feature_matrix = np.zeros((num_states, self.feat.num_features))
        for state, idx in self.feat.state_to_index.items():
            feature_matrix[idx] = self.feat.feature_vector(state)
        
        return torch.tensor(feature_matrix, dtype=torch.float32).cuda()
    
# class StateMappingStrategies:
#     def __init__(self, feat, state_to_index, known_states=None):
#         self.feat = feat
#         self.state_to_index = state_to_index
#         self.known_states = known_states if known_states is not None else list(state_to_index.keys())
#         self.kdtree = cKDTree(self.known_states)

#     def original_feature_matrix(self, trajectories):
#         n_features = len(self.feat.feature_vector(trajectories[0][0][0]))
#         f_matrix = np.zeros((len(self.state_to_index), n_features))

#         for trajectory in trajectories:
#             for state, _ in trajectory:
#                 if state not in self.state_to_index:
#                     print(f"Warning: State {state} not found in state_to_index. Skipping.")
#                     continue
#                 state_index = self.state_to_index[state]
#                 f_matrix[state_index] = self.feat.feature_vector(state)

#         return f_matrix

#     def nearest_neighbor_feature_matrix(self, trajectories):
#         n_features = len(self.feat.feature_vector(trajectories[0][0][0]))
#         f_matrix = np.zeros((len(self.state_to_index), n_features))

#         for trajectory in trajectories:
#             for state, _ in trajectory:
#                 if state not in self.state_to_index:
#                     _, index = self.kdtree.query(state)
#                     nearest_state = tuple(self.known_states[index])
#                     print(f"Warning: Mapping unseen state {state} to nearest known state {nearest_state}")
#                     state = nearest_state
#                 state_index = self.state_to_index[state]
#                 f_matrix[state_index] = self.feat.feature_vector(state)

#         return f_matrix

#     def interpolation_feature_matrix(self, trajectories, k=3):
#         n_features = len(self.feat.feature_vector(trajectories[0][0][0]))
#         f_matrix = np.zeros((len(self.state_to_index), n_features))

#         for trajectory in trajectories:
#             for state, _ in trajectory:
#                 if state not in self.state_to_index:
#                     distances, indices = self.kdtree.query(state, k=k)
#                     weights = 1 / (distances + 1e-6)
#                     weights /= np.sum(weights)
#                     feature_vec = np.zeros(n_features)
#                     for idx, weight in zip(indices, weights):
#                         known_state = tuple(self.known_states[idx])
#                         feature_vec += weight * self.feat.feature_vector(known_state)
#                     print(f"Interpolating feat for unseen state {state}")
#                 else:
#                     state_index = self.state_to_index[state]
#                     feature_vec = self.feat.feature_vector(state)
                
#                 f_matrix[state_index] = feature_vec

#         return f_matrix

#     def state_abstraction_feature_matrix(self, trajectories, n_digits=2):
#         n_features = len(self.feat.feature_vector(trajectories[0][0][0]))
#         f_matrix = np.zeros((len(self.state_to_index), n_features))

#         def abstract_state(state):
#             return tuple(round(s, n_digits) for s in state)

#         for trajectory in trajectories:
#             for state, _ in trajectory:
#                 abstract_s = abstract_state(state)
#                 if abstract_s not in self.state_to_index:
#                     print(f"Warning: Abstract state {abstract_s} not in state_to_index. Skipping.")
#                     continue
#                 state_index = self.state_to_index[abstract_s]
#                 f_matrix[state_index] = self.feat.feature_vector(state)

#         return f_matrix

#     def get_feature_matrix(self, method, trajectories, **kwargs):
#         if method == 'original':
#             return self.original_feature_matrix(trajectories)
#         elif method == 'nearest_neighbor':
#             return self.nearest_neighbor_feature_matrix(trajectories)
#         elif method == 'interpolation':
#             k = kwargs.get('k', 3)
#             return self.interpolation_feature_matrix(trajectories, k)
#         elif method == 'state_abstraction':
#             n_digits = kwargs.get('n_digits', 2)
#             return self.state_abstraction_feature_matrix(trajectories, n_digits)
#         else:
#             raise ValueError(f"Unknown method: {method}")

if __name__ == '__main__':
    n_pos = 20
    n_vel = 20
    n_f = 20

    discretised_demos = DM.data_manip(n_pos, n_vel, n_f, 'data_extracted', 'data_discretised')[0]
    trajectories = MDP.create_trajectories(discretised_demos, 'trajectories')
    states = trajectories.values()
    state = list(states)[0][-1][0]
    print(f"state: {state}")
    feat = Features(trajectories, 20)
    features_vec = feat.feature_vector(state)
    print(f"length of the feature vector: {len(features_vec)}")
    print(features_vec)
    state_mapping = StateMappingStrategies(feat)

    state_index_mapping = feat.state_to_index
    idx = state_index_mapping.get(state)

    feature_matrix = state_mapping.get_feature_matrix('original')
    print(feature_matrix.shape)
    # print first row of feature matrix, all the columns
    print(feature_matrix[idx, :])


