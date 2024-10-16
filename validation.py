from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats

def plot_reward_distributions(successful_rewards, unsuccessful_rewards):
    # Prepare data
    successful = [r for r in successful_rewards.values() if isinstance(r, (int, float))]
    unsuccessful = [r for r in unsuccessful_rewards.values() if isinstance(r, (int, float))]
    
    data = pd.DataFrame({
        'Reward': successful + unsuccessful,
        'Type': ['Successful'] * len(successful) + ['Unsuccessful'] * len(unsuccessful)
    })
    
    # Create violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Type', y='Reward', data=data)
    plt.title('Distribution of Rewards for Successful and Unsuccessful Trajectories')
    plt.savefig('reward_distribution.png')
    plt.close()
    
    # Create box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Type', y='Reward', data=data)
    plt.title('Box Plot of Rewards for Successful and Unsuccessful Trajectories')
    plt.savefig('reward_boxplot.png')
    plt.close()

# ROC analysis
def perform_roc_analysis(successful_rewards, unsuccessful_rewards):
    # Combine rewards and create labels
    all_rewards = successful_rewards + unsuccessful_rewards
    labels = [1] * len(successful_rewards) + [0] * len(unsuccessful_rewards)
    
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(labels, all_rewards)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    return roc_auc

# feature importance analysis
def analyze_feature_importance(theta_list, feature_names,n_features=12,n_bins=20, n_ohe_features=10):

    aggregated_importance = np.zeros((len(theta_list), n_features))
    
    for i, theta in enumerate(theta_list):
        if theta is not None:
            theta_np = theta.cpu().numpy()
            # Aggregate importance for one-hot encoded features
            for j in range(n_ohe_features):
                start_idx = j * n_bins
                end_idx = (j + 1) * n_bins
                aggregated_importance[i, j] = np.sum(np.abs(theta_np[start_idx:end_idx]))
            # Add importance for the last two features
            aggregated_importance[i, n_ohe_features:] = np.abs(theta_np[-2:])
    
    plt.figure(figsize=(12, 6))
    x = np.arange(n_features)
    width = 0.25
    
    for i in range(len(theta_list)):
        plt.bar(x + i*width, aggregated_importance[i], width, label=f'Subtask {i}')
    
    plt.xlabel('Features')
    plt.ylabel('Aggregated Absolute Theta Value')
    plt.title('Feature Importance Across Subtasks')
    plt.xticks(x + width, feature_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

#statistical tests to determine if there's a significant difference between the rewards of successful and unsuccessful trajectories
def analyze_reward_differentiation(successful_rewards, unsuccessful_rewards):
    # Prepare data
    successful = [r for r in successful_rewards.values() if isinstance(r, (int, float))]
    unsuccessful = [r for r in unsuccessful_rewards.values() if isinstance(r, (int, float))]
    
    # Perform Mann-Whitney U test
    statistic, p_value = stats.mannwhitneyu(successful, unsuccessful, alternative='two-sided')
    
    # Calculate effect size (Cohen's d)
    mean_diff = np.mean(successful) - np.mean(unsuccessful)
    pooled_std = np.sqrt((np.std(successful)**2 + np.std(unsuccessful)**2) / 2)
    effect_size = mean_diff / pooled_std
    
    print(f"Mann-Whitney U test statistic: {statistic}")
    print(f"p-value: {p_value}")
    print(f"Effect size (Cohen's d): {effect_size}")
    
    if p_value < 0.05:
        print("The difference in rewards between successful and unsuccessful trajectories is statistically significant.")
    else:
        print("There is no statistically significant difference in rewards between successful and unsuccessful trajectories.")
    
    print(f"The effect size indicates a {'small' if abs(effect_size) < 0.5 else 'medium' if abs(effect_size) < 0.8 else 'large'} practical significance.")

# treat this as a binary classification problem and calculate relevant metrics: will quantify the algorithm's discriminative power.
def calculate_classification_metrics(successful_rewards, unsuccessful_rewards):
    # Prepare data
    successful = [r for r in successful_rewards.values() if isinstance(r, (int, float))]
    unsuccessful = [r for r in unsuccessful_rewards.values() if isinstance(r, (int, float))]
    
    all_rewards = successful + unsuccessful
    true_labels = [1] * len(successful) + [0] * len(unsuccessful)
    
    # Calculate threshold (you might want to experiment with different thresholds)
    threshold = np.mean(all_rewards)
    
    # Predict based on threshold
    predicted_labels = [1 if r >= threshold else 0 for r in all_rewards]
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    
    print(f"Classification Metrics (threshold = {threshold:.2f}):")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")
