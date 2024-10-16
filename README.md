# Temporally Contextual Maximum Causal Entropy Inverse Reinforcement Learning

This repository contains the code for my Master's thesis on Temporally Contextual Maximum Causal Entropy (TC-MCE) Inverse Reinforcement Learning (IRL) for robotic manipulation tasks. The primary goal of this project is to learn a reward function from expert demonstrations that accurately captures the desired behavior for successful task completion, specifically focusing on grasping and lifting objects.

## Key Features

* **Temporal Context:** Unlike traditional IRL methods that rely on the Markov property, this code incorporates temporal context by considering a history of past states. This allows the algorithm to capture time-dependent aspects of the task, such as force impulse and spatial positioning, which are crucial for successful grasping. This is achieved by employing a sliding window approach that maintains a history of up to 11 states (the current state plus ten previous states) along the discredited demonstration.
* **Subtask Detection:** The code automatically divides the grasping task into distinct phases (subtasks) using change point detection algorithms. This enables the learning of separate reward functions for each subtask, providing a more nuanced understanding of the overall task structure. The subtask detection process uses the Pruned Exact Linear Time (PELT) algorithm, which is applied to a transformed representation of the extracted and filtered data. This representation encodes each of the sampled variables as a time series of the features force-velocity ratio (f/v) and the binary lift indicator.
* **Maximum Causal Entropy IRL:** The core of the code is the MCE IRL algorithm, which learns a reward function by maximizing the causal entropy of the policy while matching feature expectations from expert demonstrations. This approach addresses the ambiguity inherent in IRL problems by selecting the distribution over trajectories with the highest entropy. The algorithm begins with an initialization phase, where for each subtask k, reward weights $\theta_k$ are initialized as small random values, in the order of $10^{-4}$. A convergence threshold of $\epsilon = 10^{-4}$ is defined along with the learning rate. Following initialization, the algorithm computes the expert feature expectations $\mu_{E,k}$ for each subtask.
* **Data Processing and Visualization:** The code includes comprehensive tools for data manipulation, including extraction, filtering, discretization, and plotting. This facilitates data preparation and analysis for the IRL algorithm. The data processing tools include functions for extracting relevant variables from raw data files, applying sub-sampling and low-pass filters to reduce data volume and smooth velocity data, and discretizing continuous variables into bins. The visualization tools include functions for generating plots for individual demonstrations, visualizing the changes in variables over time and highlighting subtask boundaries.

## How it Works

1. **Data Collection:** Expert demonstrations of grasping and lifting tasks are collected using a bilateral control system with a master and slave robot.
2. **Data Preprocessing:** The collected data is filtered, sub-sampled, and discretized to prepare it for the IRL algorithm. This is handled by the `data_manip.py` file, which includes functions for extracting relevant variables, applying filters, and discretizing the data.
3. **Subtask Detection:** The continuous demonstrations are segmented into subtasks using change point detection techniques applied to features like force-velocity ratio and lift indicator. This is implemented in the `subtask_detection.py` file, which utilizes the PELT algorithm for change point detection.
4. **Trajectory Creation:** Trajectories are created by incorporating temporal context using a sliding window approach, maintaining a history of past states.
5. **Feature Extraction:** Relevant features are extracted from the trajectories to describe the state space, including dynamic features like force impulse, velocity smoothness, and subtask transition probabilities. The `features.py` file contains functions for calculating these features.
6. **MCE IRL:** The MCE IRL algorithm is applied to learn subtask-specific reward functions that explain the expert demonstrations. This is the core functionality of the `mce_irl_torch.py` file, which implements the MCE IRL algorithm using PyTorch.
7. **Evaluation:** The learned reward functions are evaluated based on their ability to distinguish between successful and unsuccessful grasping trajectories using metrics like ROC AUC, classification accuracy, and statistical significance tests. The `validation.py` file provides tools for evaluating the performance of the learned reward functions.

## Dependencies

* Python 3.x
* NumPy
* SciPy
* Pandas
* Matplotlib
* PyTorch


## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues if you find any bugs or have suggestions for improvements.

## License

This project is licensed under the MIT License.
