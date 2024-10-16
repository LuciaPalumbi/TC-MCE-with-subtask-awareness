##!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from ruptures import Pelt
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def fv_ratio_calc(force, velocity, eps = 1e-4, smoothing = False):
    fv_ratio = np.where(np.abs(velocity) > eps, 
                        np.array(force) / np.array(velocity), 
                        np.array(force) / (np.sign(velocity) * eps)
                        ) # 
    #fv_ratios = np.log(np.abs(force) + eps) / np.log(np.abs(velocity) + eps)

    if smoothing:
        fv_ratio = np.convolve(fv_ratio, np.ones(15)/15, mode='same')
    return fv_ratio

def load_and_preprocess_data(data_folder, window = 10, fv_reprocess = 'log'):
    trajectories = {}
    for filename in os.listdir(data_folder):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(data_folder, filename))
            trajectory = df.values

            # Add normalized time feature
            #time_feature = np.linspace(0, 1, len(trajectory))
            # add a non normalised time feature
            time_feature = np.arange(len(trajectory))
              
            position = trajectory[:, 1]
            velocity = trajectory[:, 2]
            force = trajectory[:, 3]
            lift = trajectory[:, 0]
            
            # Calculate force/velocity ratio
            f_v_ratio = fv_ratio_calc(force, velocity)
            if fv_reprocess == 'log':
                #log transformation
                f_v_ratio = np.log1p(np.abs(f_v_ratio)) * np.sign(f_v_ratio) #  Take the logarithm of the absolute values to reduce extreme spikes.
            elif fv_reprocess == 'smoothing':
                # Smoothing
                f_v_ratio = np.log1p(np.abs(f_v_ratio)) * np.sign(f_v_ratio) #  Take the logarithm of the absolute values to reduce extreme spikes.
                f_v_ratio = np.convolve(f_v_ratio, np.ones(window)/window, mode='same') # Apply a moving average or exponential smoothing to reduce noise
            elif fv_reprocess == 'clip':
                # Clip extreme values
                clip_threshold = np.percentile(np.abs(f_v_ratio), 99)
                f_v_ratio = np.clip(f_v_ratio, -clip_threshold, clip_threshold) # Set a threshold to cap very large spikes.

            # boolean lifting features
            for idx, l in enumerate(lift):
                if l < 3.3:
                    lift[idx] = 0
                else:
                    lift[idx] = 5
            
            enhanced_trajectory = np.column_stack((f_v_ratio, lift, time_feature))
            #enhanced_trajectory = f_v_ratio.reshape(-1, 1)

            trajectories[filename] = enhanced_trajectory
    
    return trajectories

class MultivariateSubtaskDetector:
    def __init__(self, max_subtasks=4):
        self.max_subtasks = max_subtasks


class SubtaskDetector:
    def __init__(self, max_subtasks=4):
        self.max_subtasks = max_subtasks

    def detect_subtasks(self, trajectories, method='pelt', reverse=True):
        if method == 'kmeans':
            return self.kmeans_subtask_detection(trajectories)
        elif method == 'pelt':
            return self.pelt_subtask_detection(trajectories, reverse = True)
        elif method == 'multivariate':
            return self.multivariate_pelt_detection(trajectories, reverse = True)
        elif method == 'spectral':
            return self.spectral_subtask_detection(trajectories)
        elif method == 'hybrid':
            return self.hybrid_subtask_detection(trajectories)
        elif method == 'ordered_kmeans':
            return self.ordered_kmeans_subtask_detection(trajectories)
        elif method == 'random_forest':
            return self.random_forest_subtask_detection(trajectories)
        else:
            raise ValueError("Unknown method. Choose 'kmeans', 'pelt', 'spectral', 'hybrid', 'random forest' or 'ordered_kmeans'.")

    def determine_optimal_n_clusters(self, data): # elbow method
        inertias = []
        k_range = range(1, self.max_subtasks + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
        
        # Calculate the differences in inertia
        inertia_differences = np.diff(inertias)
        
        # Find the elbow point (where the difference starts to level off)
        elbow_point = np.argmax(inertia_differences) + 1
        
        # Ensure we have at least 2 clusters
        return max(2, min(elbow_point + 1, self.max_subtasks))
    
    def consensus_subtasks(self, trajectories):
        # First pass: Apply PELT to each trajectory
        change_points_per_trajectory = []
        for traj in trajectories.values():
            model = Pelt(model="rbf").fit(traj)
            change_points = [len(model.predict(pen=p)) - 1 for p in range(10, 60, 5)] # Try different penalties
            change_points_per_trajectory.append(change_points)
        
        # Aggregate results
        all_change_points = [cp for traj_cps in change_points_per_trajectory for cp in traj_cps]
        
        # Consensus building
        mode_change_points = max(set(all_change_points), key=all_change_points.count)
        
        # Optional: Clustering on change point locations
        # This part is more complex and might require additional libraries
        
        return mode_change_points


    def pelt_subtask_detection(self, trajectories, reverse):
        subtask_labels = []
        demos_labels = []
        n_change_points = self.consensus_subtasks(trajectories)
        print(f"Consensus number of change points: {n_change_points}")

        for filename, traj in trajectories.items():   
            if reverse:
                traj_length = len(traj)
                traj_rev = traj[::-1]
                min_penalty=1 
                max_penalty=100
                # Apply PELT algorithm
                for _ in range(50):
                    penalty = (min_penalty + max_penalty) / 2
                    model = Pelt(model="rbf").fit(traj_rev[:, :-1])
                    change_points_reversed = model.predict(pen=penalty)
                    if len(change_points_reversed) - 1 == n_change_points:
                        break
                    elif len(change_points_reversed) - 1 > n_change_points:
                        min_penalty = penalty
                    else:
                        max_penalty = penalty

                change_points = [traj_length - 1 - cp for cp in change_points_reversed]
                # reorder change points to be in ascending order
                change_points_sorted = sorted(change_points)
                demo_labels = np.zeros(len(traj), dtype=int)
                # Assign labels based on change points
                #print(f"Change points: {change_points_sorted}")

                for i in range(len(demo_labels)):
                    if i < change_points_sorted[1]:
                        demo_labels[i] = 0
                    elif len(change_points) == 2 and change_points_sorted[1] <= i: 
                        demo_labels[i] = 1
                    elif len(change_points) == 3 and change_points_sorted[1] <= i < change_points_sorted[2]: 
                        demo_labels[i] = 1
                    elif len(change_points) == 3 and i >= change_points_sorted[2]:
                            demo_labels[i] = 2
            
            # Create labels
            labels = np.zeros(len(traj), dtype=int)
            for i, cp in enumerate(change_points[:-1]):
                labels[cp:] = i + 1

            subtask_labels.append(labels)
            
            # Plotting
            fig, axs = plt.subplots(traj.shape[1] - 1, 1, figsize=(15, 5 * (traj.shape[1] - 1)), sharex=True)
            fig.suptitle(f'{filename} with Change Points', fontsize=16)
            
            time = traj[:, -1]  # Time feature
            feature_names = ['F/V Ratio', 'Lift']
            
            for i in range(traj.shape[1] - 1):
                axs[i].plot(time, traj[:, i], label=feature_names[i])
                axs[i].set_ylabel(feature_names[i])
                
                # Add vertical lines for change points
                for cp in change_points[:-1]:
                    axs[i].axvline(x=time[cp], color='r', linestyle='--', alpha=0.5)
                
                axs[i].legend()
            
            axs[-1].set_xlabel('Time')
            plt.tight_layout()
            plt.savefig(f'pelt_{filename}_{"reversed" if reverse else "forward"}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
            demos_labels.append(demo_labels)
        print(f'length of demo labels: {len(demos_labels[0])}')
        return demos_labels

    def compute_score(self, data, change_points):
        # Implement a scoring function, in this case a combination of RSS and a penalty for the number of change points
        n_samples, n_dims = data.shape
        rss = 0
        for start, end in zip([0] + change_points[:-1], change_points):
            segment = data[start:end]
            rss += np.sum((segment - np.mean(segment, axis=0))**2)
        
        # Penalty term (you might need to adjust this)
        penalty = len(change_points) * np.log(n_samples) * n_dims
        
        return rss + penalty
    
    def find_optimal_n_change_points(self, data, min_n=2, max_n=4,min_penalty=1, max_penalty=100):
        model = Pelt(model="rbf").fit(data)
        n_samples = data.shape[0]
        
        best_n = min_n
        best_score = float('inf')
        best_penalty = min_penalty
        
        for penalty in np.logspace(np.log10(min_penalty), np.log10(max_penalty), num=20):
            change_points = model.predict(pen=penalty)
            n_bkps = len(change_points) - 1  # number of change points
            
            if min_n <= n_bkps <= max_n:
                score = self.compute_score(data, change_points)
                
                if score < best_score:
                    best_score = score
                    best_n = n_bkps
                    best_penalty = penalty
    
        return best_n
    
    def find_optimal_n_change_points_for_all(self, trajectories):
        optimal_ns = []
        for traj in trajectories.values():
            optimal_n = self.find_optimal_n_change_points(traj)
            optimal_ns.append(optimal_n)
        
        # Use median as the consensus
        return int(np.median(optimal_ns))
    
    def multivariate_pelt_detection(self, trajectories, reverse=True):
        subtask_labels = []
        demos_labels = []
        all_results = []

        for filename, traj in trajectories.items():
            if reverse:
                traj_rev = traj[::-1]
            
            # Exclude the time feature
            data = traj_rev[:, :-1]
            
            # Determine optimal number of change points
            n_change_points = self.find_optimal_n_change_points_for_all(trajectories)

            min_penalty=1 
            max_penalty=100
            # Apply PELT algorithm
            for _ in range(50):
                penalty = (min_penalty + max_penalty) / 2
                model = Pelt(model="rbf").fit(traj_rev[:, :-1])
                change_points_reversed = model.predict(pen=penalty)
                if len(change_points_reversed) - 1 == n_change_points:
                    break
                elif len(change_points_reversed) - 1 > n_change_points:
                    min_penalty = penalty
                else:
                    max_penalty = penalty
            
            # Create labels
            labels = np.zeros(len(traj), dtype=int)
            for i, cp in enumerate(change_points[:-1]):
                labels[cp:] = i + 1
            
            if reverse:
                labels = labels[::-1]
                change_points = [len(traj) - 1 - cp for cp in change_points][::-1]
            
            subtask_labels.append(labels)
            demos_labels.append(labels)
            
            all_results.append({
                'filename': filename,
                'traj': traj,
                'change_points': change_points,
                'reverse': reverse
            })

        # Plot all results together
        self.plot_all_results(all_results)

        return demos_labels

    def plot_all_results(self, all_results):
        n_trajectories = len(all_results)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05,
                            subplot_titles=('F/V Ratio', 'Lift'))

        colors = px.colors.qualitative.Plotly  # Use Plotly's default color sequence

        for i, result in enumerate(all_results):
            filename = result['filename']
            traj = result['traj']
            change_points = result['change_points']
            reverse = result['reverse']

            time = traj[:, -1]  # Time feature
            color = colors[i % len(colors)]  # Cycle through colors

            # F/V Ratio plot
            fig.add_trace(go.Scatter(x=time, y=traj[:, 0], mode='lines', name=f'{filename} - F/V Ratio',
                                     line=dict(color=color)), row=1, col=1)

            # Lift plot
            fig.add_trace(go.Scatter(x=time, y=traj[:, 1], mode='lines', name=f'{filename} - Lift',
                                     line=dict(color=color)), row=2, col=1)

            # Add vertical lines for change points
            for cp in change_points[:-1]:
                fig.add_shape(type="line", x0=time[cp], y0=0, x1=time[cp], y1=1,
                              line=dict(color="red", width=1, dash="dash"),
                              xref=f'x', yref='paper')

        fig.update_layout(height=800, width=1200, title_text="All Trajectories with Change Points",
                          showlegend=True, legend_tracegroupgap=5)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="F/V Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Lift", row=2, col=1)

        fig.write_html("all_trajectories_change_points.html")
        fig.show()

#############################################################################################################

    def spectral_subtask_detection(self, trajectories):
        subtask_labels = []
        for traj in trajectories:
            affinity = self._compute_affinity(traj)
            spectral = SpectralClustering(n_clusters=self.determine_optimal_n_clusters(traj), affinity='precomputed', random_state=42)
            labels = spectral.fit_predict(affinity)
            subtask_labels.append(labels)
        return subtask_labels
    
    def random_forest_subtask_detection(self, trajectories):
        subtask_labels = []
        for traj in trajectories:
            
            # Train a Random Forest classifier
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Use the first half of the trajectory for training
            mid_point = len(trajectories) // 2
            train_data = trajectories[:mid_point]
            n_subtasks = self.determine_optimal_n_clusters(train_data)
            train_labels = np.arange(len(train_data)) // (len(train_data) // self.max_subtasks)
            
            rf.fit(train_data, train_labels)
            
            # Predict on the entire trajectory
            labels = rf.predict(trajectories)
            
            subtask_labels.append(labels)
        
        return subtask_labels
    
    def hybrid_subtask_detection(self, trajectories):
        subtask_labels = []
        for traj in trajectories:
            # 1. Use PELT for change point detection
            model = Pelt(model="rbf").fit(traj)
            change_points = model.predict(pen=10)
            
            # 2. Use the number of change points to inform K-means
            n_clusters = min(len(change_points), self.max_subtasks)
            n_clusters = max(1, n_clusters)  # Ensure at least 1 cluster
            
            # 3. Apply K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans_labels = kmeans.fit_predict(traj)
            
            # 4. Apply Spectral Clustering
            affinity_matrix = self._compute_affinity(traj)
            affinity_matrix += 1e-5 
            spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42, n_init=100, assign_labels='kmeans', eigen_tol=1e-3)
            spectral_labels = spectral.fit_predict(affinity_matrix)
            
            # 5. Combine the results
            combined_features = np.column_stack((
                kmeans_labels, 
                spectral_labels, 
                np.digitize(np.arange(len(traj)), change_points[:-1])
            ))
            
            # 6. Final clustering on combined results
            final_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            final_labels = final_kmeans.fit_predict(combined_features)
            
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

def main():
    data_folder = 'train_extracted'  # Update this to your folder path
    trajectories = load_and_preprocess_data(data_folder, window=10)

    detector = SubtaskDetector()
    
    methods = ['pelt']
    for method in methods:
        subtask_labels = detector.detect_subtasks(trajectories, method=method, reverse=True)

        print(f"Subtask detection using {method} method completed.")

if __name__ == "__main__":
    main()