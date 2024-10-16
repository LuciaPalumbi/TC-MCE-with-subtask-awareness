##!/usr/bin/env python3

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
from ruptures import Pelt
from sklearn.manifold import spectral_embedding
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import data_manip as DM
import mdp as MDP

class DescreteSubtaskDetector:
    def __init__(self, n_subtasks=3):
        self.n_subtasks = n_subtasks

    def detect_subtasks(self, trajectories, method='kmeans'):
        if method == 'kmeans':
            return self.kmeans_subtask_detection(trajectories)
        elif method == 'pelt':
            return self.pelt_subtask_detection(trajectories)
        elif method == 'spectral':
            return self.spectral_subtask_detection(trajectories)
        elif method == 'hybrid':
            return self.hybrid_subtask_detection(trajectories)
        else:
            raise ValueError("Unknown method. Choose 'kmeans', 'pelt', or 'spectral'.")

    def preprocess_data(self, trajectories):
        max_length = max(len(traj) for traj in trajectories)
        features = []
        original_lengths = []
        
        for trajectory in trajectories:
            traj_features = []
            for state, _ in trajectory:
                position, velocity, force, lift = state
                f_v_ratio = force/velocity if velocity != 0 else force/0.0001
                traj_features.append([position, velocity, force, lift, f_v_ratio])
            original_lengths.append(len(traj_features))

            # Pad the trajectory features if necessary
            padding_length = max_length - len(traj_features)
            if padding_length > 0:
                pad = np.zeros((padding_length, 5))
                traj_features = np.vstack((traj_features, pad))
            
            features.append(traj_features)
        
        features = np.array(features)
        
        # Reshape for normalization
        original_shape = features.shape
        features_flat = features.reshape(-1, features.shape[-1])
        
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_flat)
        
        # Reshape back to original shape
        features = features_normalized.reshape(original_shape)
        
        return features, original_lengths

    def kmeans_subtask_detection(self, trajectories):
        features, _ = self.preprocess_data(trajectories)
        features_flat = features.reshape(-1, features.shape[-1])
        kmeans = KMeans(n_clusters=self.n_subtasks, random_state=42)
        subtask_labels = kmeans.fit_predict(features_flat)
        subtask_labels = subtask_labels.reshape(features.shape[:-1])
        
        # Remove labels for padded zeros
        return [labels[:len(traj)] for labels, traj in zip(subtask_labels, trajectories)]

    def pelt_subtask_detection(self, trajectories):
        features, _ = self.preprocess_data(trajectories)
        subtask_labels = []
        for traj_features, traj in zip(features, trajectories):
            traj_features = traj_features[:len(traj)]  # Remove padded zeros
            model = Pelt(model="normal").fit(traj_features)  # Use "normal" instead of "gaussian"
            change_points = model.predict(pen=10)
            labels = np.zeros(len(traj_features), dtype=int)
            for i, cp in enumerate(change_points[:-1]):
                labels[cp:] = (i + 1) % self.n_subtasks
            subtask_labels.append(labels)
        return subtask_labels

    def spectral_subtask_detection(self, trajectories):
        features, _ = self.preprocess_data(trajectories)
        subtask_labels = []
        for traj_features, traj in zip(features, trajectories):
            traj_features = traj_features[:len(traj)]  # Remove padded zeros
            affinity = self._compute_affinity(traj_features)
            sc = SpectralClustering(n_clusters=self.n_subtasks, affinity='precomputed', random_state=42)
            labels = sc.fit_predict(affinity)
            subtask_labels.append(labels)
        return subtask_labels
    
    def hybrid_subtask_detection(self, trajectories):
        features, original_lengths = self.preprocess_data(trajectories)
        subtask_labels = []

        for traj_features, original_length in zip(features, original_lengths):
            
            # Remove any padding we might have added in preprocess_data
            traj_features = traj_features[:original_length]
            
            # 1. Use PELT for change point detection
            model = Pelt(model="rbf").fit(traj_features)
            change_points = model.predict(pen=10)
            
            # 2. Use the number of change points to inform K-means
            n_clusters = min(len(change_points), self.n_subtasks)
            
            # 3. Apply K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans_labels = kmeans.fit_predict(traj_features)
            
            # 4. Apply Spectral Clustering
            affinity_matrix = self._compute_affinity(traj_features)
            spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
            spectral_labels = spectral.fit_predict(affinity_matrix)
            
            # 5. Combine the results
            combined_features = np.column_stack((
                kmeans_labels, 
                spectral_labels, 
                np.digitize(np.arange(len(traj_features)), change_points[:-1])
            ))
            
            # 6. Final clustering on combined results
            final_kmeans = KMeans(n_clusters=self.n_subtasks, random_state=42)
            final_labels = final_kmeans.fit_predict(combined_features)
            
            # Ensure final_labels matches the original trajectory length
            if len(final_labels) != original_length:
                print(f"Warning: Labels length {len(final_labels)} doesn't match trajectory length {original_length}.")
                if len(final_labels) > original_length:
                    final_labels = final_labels[:original_length]
                else:
                    final_labels = np.pad(final_labels, (0, original_length - len(final_labels)), mode='edge')
            
            subtask_labels.append(final_labels)

        return subtask_labels

    def _compute_affinity(self, features):
        n_samples = len(features)
        affinity = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i, n_samples):
                sim = np.exp(-np.sum((features[i] - features[j])**2))
                affinity[i, j] = affinity[j, i] = sim
        return affinity

    def analyze_subtasks(self, trajectories, subtask_labels):
        for traj_idx, (trajectory, labels) in enumerate(zip(trajectories, subtask_labels)):
            print(f"Trajectory {traj_idx + 1}:")
            for subtask in range(self.n_subtasks):
                subtask_states = [state for (state, _), label in zip(trajectory, labels) if label == subtask]
                if subtask_states:
                    avg_position = np.mean([state[0] for state in subtask_states])
                    avg_velocity = np.mean([state[1] for state in subtask_states])
                    avg_force = np.mean([state[2] for state in subtask_states])
                    avg_lift = np.mean([state[3] for state in subtask_states])
                    print(f"  Subtask {subtask}:")
                    print(f"    Avg Position: {avg_position:.4f}")
                    print(f"    Avg Velocity: {avg_velocity:.4f}")
                    print(f"    Avg Force: {avg_force:.4f}")
                    print(f"    Avg Lift: {avg_lift:.4f}")
            print()

def plot_trajectories_with_subtasks(trajectories, subtask_labels, method_name):
    n_trajectories = len(trajectories)
    n_subtasks = max(max(labels) for labels in subtask_labels) + 1
    colors = plt.cm.rainbow(np.linspace(0, 1, n_subtasks))

    fig, axs = plt.subplots(n_trajectories, 5, figsize=(25, 5*n_trajectories))
    fig.suptitle(f'Trajectories with Subtasks - {method_name}', fontsize=16)

    for i, (trajectory, labels) in enumerate(zip(trajectories, subtask_labels)):
        position = [state[0] for state, _ in trajectory]
        velocity = [state[1] for state, _ in trajectory]
        force = [state[2] for state, _ in trajectory]
        lift = [state[3] for state, _ in trajectory]
        f_v_ratio = [f / (v + 1e-6) for f, v in zip(force, velocity)]
        time = range(len(trajectory))  # This ensures we use all steps

        for j, (y, title) in enumerate(zip([position, velocity, force, lift, f_v_ratio], 
                                           ['Position', 'Velocity', 'Force', 'Lift', 'Force/Velocity Ratio'])):
            for subtask in range(n_subtasks):
                mask = labels == subtask
                axs[i, j].scatter(np.array(time)[mask], np.array(y)[mask], 
                                  c=[colors[subtask]], label=f'Subtask {subtask}', s=1)  # Reduced point size
            axs[i, j].set_title(f'Trajectory {i+1} - {title}')
            axs[i, j].set_xlabel('Time')
            axs[i, j].set_ylabel(title)
            if i == 0 and j == 4:
                axs[i, j].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    fig.set_size_inches(30, 5*n_trajectories)  # Increase figure width
    plt.savefig(f'subtask_detection_{method_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Parameters
    n_p_bins = 90
    n_v_bins = 90
    n_f_bins = 90

    # Create trajectories from data
    discretised_demos, positions, velocities, forces, lifts = DM.data_manip(n_p_bins, n_v_bins, n_f_bins, 'data_extracted', 'data_discretised')
    trajectories = MDP.create_trajectories(discretised_demos, 'trajectories')

    # Initialize SubtaskDetector
    detector = DescreteSubtaskDetector(n_subtasks=3)

    # Apply all subtask detection methods
    methods = ['kmeans', 'pelt', 'spectral', 'hybrid']
    for method in methods:
        print(f"Applying {method} subtask detection...")
        subtask_labels = detector.detect_subtasks(trajectories, method=method)
        
        # Analyze subtasks
        print(f"Subtask analysis for {method}:")
        detector.analyze_subtasks(trajectories, subtask_labels)
        
        # Plot trajectories with subtasks
        plot_trajectories_with_subtasks(trajectories, subtask_labels, method)

        # Store updated trajectories (you might want to save these to a file)
        updated_trajectories = []
        for trajectory, labels in zip(trajectories, subtask_labels):
            updated_trajectory = [(state, action, label) for (state, action), label in zip(trajectory, labels)]
            updated_trajectories.append(updated_trajectory)
        
        # Here you could save updated_trajectories to a file if needed
        # Example: np.save(f'updated_trajectories_{method}.npy', updated_trajectories)

    print("Subtask detection and analysis complete. Plots have been saved.")

if __name__ == "__main__":
    main()