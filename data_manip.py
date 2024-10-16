#!/usr/bin/env python3
import os
import csv
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import signal
from datetime import datetime
from subtask_detection import SubtaskDetector, load_and_preprocess_data
import seaborn as sns
import math
import ast

def reset(folder_path): # to delete and recreate the folders with the results
    try:
        # Check if the folder exists
        if not os.path.exists(folder_path):
            print(f"The folder {folder_path} does not exist.")
            return
        # Check if the folder is empty
        if not os.listdir(folder_path):
            # end the function if the folder is empty
            return
        # Iterate over all items in the folder
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            
            # If it's a file, delete it
            if os.path.isfile(item_path):
                os.unlink(item_path)
        
    except Exception as e:
        print(f"An error occurred: {e}")

# From the raw data, saved in a .CSV, it filters the columns for joint8, position, velocity and force of the gripper
# It also applies a low pass filter to the velocity values to reduce the noise in the data
# It then saves the filtered data to a new CSV file in the output folder (extracted_data)
def extract_and_save_filtered_data(data_input_folder, filtered_data_folder, sampling_frequency):
    reset(filtered_data_folder)
    i = 1  # Counter for the number of files processed
    os.makedirs(filtered_data_folder, exist_ok=True)  # Ensures the output folder exists

    for filename in os.listdir(data_input_folder): # Iterates over all the files in the folder
        if filename.endswith(".csv"):
            file_path = os.path.join(data_input_folder, filename)
            data = pd.read_csv(file_path, header=None, dtype=float)    # Reads the CSV file into a DataFrame
            # delete last row of the data
            data = data[:-1]
            # Check if the file contains all the required columns and extract the required columns's data
            required_columns = data.columns[[35, 43, 44, 46]] if len(data.columns) > 47 else None

            if required_columns is not None:
                # Select only the required columns from the filtered rows
                filtered_data = data[required_columns]
                # Filter rows where index % 100 == 0
                filtered_data = filtered_data.iloc[::sampling_frequency]
                last_row = data[required_columns].iloc[-1:]
                filtered_data = pd.concat([filtered_data, last_row], ignore_index=True)

                #apply low pass filter to velocity values to reduce the noise in the data
                velocity = filtered_data.iloc[:,2]
                order = 2       # order of the filter
                sampling_freq = 10000  # 100 Hz sampling rate (adjust if different)
                cutoff_freq = 100 # cutoff frequency in Hz
                nyquist_freq = 0.5 * sampling_freq
                normalized_cutoff = cutoff_freq / nyquist_freq
                # Create the low-pass Butterworth filter
                b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
                filtered_velocity = signal.filtfilt(b, a, velocity)
                # Replace the velocity values in the DataFrame with the filtered values
                filtered_data.iloc[:,2] = filtered_velocity

                # extract the column related to the force and turn all values > 0 to 0
                filtered_data.iloc[:, 3] = np.where(filtered_data.iloc[:, 3] > 0, 0, filtered_data.iloc[:, 3])

                # Save the filtered data to a new CSV file
                file_name, file_extension = os.path.splitext(filename)
                new_file_name = f"{file_name}_data{file_extension}"
                output_file_path = os.path.join(filtered_data_folder, os.path.basename(new_file_name))
                filtered_data.to_csv(output_file_path, index=False)
                i += 1
            else:
                print(f"File {filename} does not contain all required columns.")

    if i == 1:  # No files were processed
        print(f"Error: No suitable CSV files found in the folder: {data_input_folder}")

# Create a list of lists, each containing the data from one of the demonstrations files contained in the filtered_data_path
def create_demos_list(filtered_data_path):
    """
    Reads multiple CSV files from a given folder path and creates a list of lists, each containing the data from 
    one of the demonstrations files.
    The data for each demonstration is made up of a list of tuples, each containing the values of position, velocity, force and lift of each file
    """
    demos = []
    raw_data = []
    csv_files_found = False

    # Iterate over all the files in the folder
    for filename in os.listdir(filtered_data_path):
        if filename.endswith(".csv"):
            # j = 0                                  # for debugging
            csv_files_found = True
            file_path = os.path.join(filtered_data_path, filename)
            data_list = []
            
            # Open the CSV file and read its contents
            with open(file_path, 'r') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)                     # Skip the header row
                # Iterate over each row in the CSV file
                for row in csv_reader:
                    tuple_data = (float(row[1]), float(row[2]), float(row[3]), float(row[0]))
                    data_list.append(tuple_data)                 
            demos.append(data_list)
            raw_data.append(np.array(data_list))

        if not csv_files_found:
            print(f"Error: No CSV files found in the folder: {filtered_data_path}")

    return demos

# Find the minimum and maximum values for each variable among all the demonstrations
def find_variable_ranges(demos):
    """ takes the all_data list as input, goes through all the lists and tuples, and 
    finds the minimum and maximum values for each variable (columns 41, 42, and 43 from before)"""

    # Initialize variables to store the minimum and maximum values for each variable
    min_position = float('inf')  # Initialize with positive infinity
    max_position = float('-inf')  # Initialize with negative infinity
    min_velocity = float('inf')
    max_velocity = float('-inf')
    min_force = float('inf')
    max_force = float('-inf')
    min_lift = float('inf')
    max_lift = float('-inf')

    # Iterate over each list in discretised_demos
    for data_list in demos:
        # Iterate over each tuple in the current list
        for tuple_data in data_list:
            # Extract the values from the tuple
            position, velocity, force, lift = tuple_data

            # Convert the values to floats for comparison
            position = float(position)
            velocity = float(velocity)
            force = float(force)
            lift = float(lift)

            # Update the minimum and maximum values for position
            
            min_position = min(min_position, position)
            max_position = max(max_position, position)

            # Update the minimum and maximum values for velocity
            min_velocity = min(min_velocity, velocity)
            max_velocity = max(max_velocity, velocity)

            # Update the minimum and maximum values for force
            min_force = min(min_force, force)
            max_force = max(max_force, force)

            # Update the minimum and maximum values for lift
            min_lift = min(min_lift, lift)
            max_lift = max(max_lift, lift)

    #min_position = round(min_position, 4)
    #max_position = round(max_position, 4)
    min_position = 0
    max_position = 1
    min_velocity = round(min_velocity, 4)
    max_velocity = round(max_velocity, 4)
    min_force = round(min_force, 6)
    max_force = round(max_force, 6)
    min_lift = round(min_lift, 4)
    max_lift = round(max_lift, 4)

    # Create a dictionary to store the minimum and maximum values for each variable
    ranges = {
        'position range': (min_position, max_position),
        'velocity range': (min_velocity, max_velocity),
        'force range': (min_force, max_force),
        'lift range': (min_lift, max_lift)
    }
    return ranges

# Discretisation:

# Discretize the demonstrations data into bins for position, velocity, force and lift
def discretise_demos(filtered_data_folder, ranges, n_p_bins, n_v_bins, n_f_bins, discretised_demos_folder,detector_method='pelt', n_lift_bins = 3):
    
    reset(discretised_demos_folder)
    detector = SubtaskDetector()
    trajectories = load_and_preprocess_data(filtered_data_folder, window=10, fv_reprocess = 'log')
    subtask_labels = detector.detect_subtasks(trajectories, detector_method, reverse = True)

    # Initialize an empty list to store the discretised demonstrations
    discretised_demos = {}
    # Calculate the step size for each variable
    p_min, p_max = ranges['position range']
    v_min, v_max = ranges['velocity range']
    f_min, f_max = ranges['force range']
    lift_min, lift_max = ranges['lift range']
    p_step = round((p_max - p_min) / (n_p_bins - 1), 6)
    v_step = round((v_max - v_min) / (n_v_bins - 1), 6)
    f_step = round((f_max - f_min) / (n_f_bins - 1), 6)
    lift_step = round((lift_max - lift_min) / n_lift_bins, 4)

    positions_round = []
    velocities_round = []
    forces_round = []
    positions = np.linspace(p_min, p_max, n_p_bins).tolist()
    velocities = np.linspace(v_min, v_max, n_v_bins).tolist()
    forces = np.linspace(f_min, f_max, n_f_bins).tolist()
    lifts = []

    
    for idx, filename in enumerate(os.listdir(filtered_data_folder)):     # Iterate over each demonstration
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(filtered_data_folder, filename))
            demo = df.values                              # Extract the values from the DataFrame
            discretised_demo = []
            subtask_label = subtask_labels[idx]

            # Iterate over each row in the demonstration
            for j, row in enumerate(demo):
                position, velocity, force, lift = row[1], row[2], row[3], row[0]
                
                # Discretize the position value
                p_value = round(min(positions, key=lambda x: abs(x - position)), 6)
                if p_value not in positions_round:
                    positions_round.append(p_value)

                # Discretize the velocity value
                v_value = round(min(velocities, key=lambda x: abs(x - velocity)), 6)
                if v_value not in velocities_round:
                    velocities_round.append(v_value)

                # Discretize the force value
                f_value = round(min(forces, key=lambda x: abs(x - force)), 6)
                if f_value > 0:
                    f_value = 0
                if f_value not in forces_round:
                    forces_round.append(f_value)

                # Discretize the lift value
                if lift < 3.4:
                    lift_value = round(lift_min + lift_step/2, 4)
                # elif lift < 2:
                #     lift_value = round(lift_min + lift_step + lift_step/2, 4)
                else:
                    lift_value = round(lift_max - lift_step/2, 4)
                if lift_value not in lifts:
                    lifts.append(lift_value)
                
                label = subtask_label[j]  
                # Create a new tuple with the discretized values
                discretised_tuple = (p_value, v_value, f_value, lift_value, label)
                discretised_demo.append(discretised_tuple)

            # Create and save a csv file with the discretised data
            file_name, file_extension = os.path.splitext(filename)
            with open(f'{discretised_demos_folder}/{file_name}_discrete.csv', 'w') as file:
                writer = csv.writer(file)
                for tuple in enumerate(discretised_demo):      
                    writer.writerow(tuple)
            
            discretised_demos[filename] = discretised_demo
        
    positions_round.sort()
    velocities_round.sort()
    forces_round.sort()
    lifts.sort()
    print('------------------------------------------')
    print("from discretise_demos:")
    print("Step Sizes:")
    print(f"Position Step: {p_step}")
    print(f"Velocity Step: {v_step}")
    print(f"Force Step: {f_step}")
    print(f"Lift Step: {lift_step}")
    print('------------------------------------------')
    print (f"number of positions: {len(positions_round)}")
    print (f"number of velocities: {len(velocities_round)}")
    print (f"number of forces: {len(forces_round)}")
    print('------------------------------------------')
   # print(f"positions: {positions_round}")
   # print('------------------------------------------')
   # print(f"velocities: {velocities_round}")
   # print('------------------------------------------')
   # print(f"forces: {forces_round}")
   # print('------------------------------------------')
   # print(f"lifts: {lifts}")
   # print('------------------------------------------')
    
    return discretised_demos, positions_round, velocities_round, forces_round, lifts

def data_manip(n_p_bins, n_v_bins, n_f_bins, filtered_data_path, discretised_demos_folder):

    demos = create_demos_list(filtered_data_path)

    print(f"Number of expert demonstrations: {len(demos)}")
    print("------------------------------------------")
    
    # Find variable ranges
    ranges = find_variable_ranges(demos)
    print("Variables Ranges:")
    print(f"Position Range: {ranges['position range']}")
    print(f"Velocity Range: {ranges['velocity range']}")
    print(f"Force Range: {ranges['force range']}")
    print(f"Lift Range: {ranges['lift range']}")
    print("------------------------------------------")

    # Call the discretise_demos function to discretize the demos
    discretised_demos, positions, velocities_round, forces_round, lifts = discretise_demos(filtered_data_path, ranges, n_p_bins, n_v_bins, n_f_bins, discretised_demos_folder)

    return discretised_demos, positions, velocities_round, forces_round, lifts

######################################################################################## Plotting #####################################################################################

def plot_single_demo(demo_path):
    file_name, file_extension = os.path.splitext(demo_path)
    
    # Read the data from the CSV file
    with open(demo_path, "r") as file:
        csv_reader = csv.reader(file)
        data = [row for row in csv_reader if row]  # Remove empty rows
   
    # Print debug information
    # print("First few rows of data:")
    # for row in data[:5]:
    #     print(row)
    
    # Extract the values
    timesteps = range(len(data))
    try:
        # Parse the tuple-like string into actual values
        parsed_data = [ast.literal_eval(row[1]) for row in data]
        position_values = [d[0] for d in parsed_data]
        velocity_values = [d[1] for d in parsed_data]
        force_values = [d[2] for d in parsed_data]
        lift_values = [d[3] for d in parsed_data]
        subtask_values = [d[4] for d in parsed_data]
    except (ValueError, IndexError) as e:
        print(f"Error parsing values: {e}")
        #print("Problematic row:", row)
        return
    
    # Calculate F/V ratio
    fv_ratio = np.where(np.abs(velocity_values) > 1e-3, 
                        np.array(force_values) / np.array(velocity_values), 
                        np.array(force_values) / (np.sign(velocity_values) * 1e-3))
   
    # Find subtask change points
    subtask_changes = [i for i in range(1, len(subtask_values)) if subtask_values[i] != subtask_values[i-1]]

    # Plot the data
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 20))
    fig_name = file_name.split("/")[-1]
    fig.canvas.manager.set_window_title(f"Demo: {fig_name}")
    fig.suptitle(f"Demo: {fig_name}", fontsize=16)

    # Function to plot data with subtask change lines
    def plot_with_subtask_changes(ax, values, label, title):
        ax.plot(timesteps, values, label=label)
        ax.set_ylabel(label)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add vertical lines for subtask changes
        for change_point in subtask_changes:
            ax.axvline(x=change_point, color='r', linestyle='--', alpha=0.5)

    # Force plot
    plot_with_subtask_changes(ax1, force_values, "Force", "Force Plot")
    
    # Velocity plot
    plot_with_subtask_changes(ax2, velocity_values, "Velocity", "Velocity Plot")
    
    # Position plot
    plot_with_subtask_changes(ax3, position_values, "Position", "Position Plot")
    
    # F/V Ratio plot
    plot_with_subtask_changes(ax4, fv_ratio, "F/V Ratio", "Force/Velocity Ratio Plot")

    plt.tight_layout()

def fv_ratio_calc(force, velocity, eps = 1e-4, smoothing = False):
    fv_ratio = np.where(np.abs(velocity) > eps, 
                        np.array(force) / np.array(velocity), 
                        np.array(force) / (np.sign(velocity) * eps))
    #fv_ratios = np.log(np.abs(force) + eps) / np.log(np.abs(velocity) + eps)
    fv_ratio = np.log1p(np.abs(fv_ratio)) * np.sign(fv_ratio) #  Take the logarithm of the absolute values to reduce extreme spikes.

    if smoothing:
        fv_ratio = np.convolve(fv_ratio, np.ones(15)/15, mode='same')

    return fv_ratio

def plot_data(data_input_folder):
     # Initialize empty lists to store the data for the original demos
    position_data = []
    velocity_data = []
    force_data = []
    lift_data = [] 
    f_v_ratios = []

    for filename in os.listdir(data_input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(data_input_folder, filename)
            
            # Read the data from the CSV file
            with open(file_path, "r") as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # Skip the header row if present
                data = list(csv_reader)
            
            # Extract the values from the columns
            position_values = [float(row[1]) for row in data]
            velocity_values = [float(row[2]) for row in data]
            force_values = [float(row[3]) for row in data]
            lift_values = [float(row[0]) for row in data]  
            f_v_ratio = fv_ratio_calc(force_values, velocity_values)
            #f_v_ratio = [np.log(np.abs(float(row[3])) + 1e-6) / np.log(np.abs(float(row[2])) + 1e-6) for row in data]
            
            # Store the data for each file
            position_data.append((filename, position_values))
            velocity_data.append((filename, velocity_values))
            force_data.append((filename, force_values))
            lift_data.append((filename, lift_values))
            f_v_ratios.append((filename, f_v_ratio))
        
    def create_plot(fig, axs, data, title, y_labels):
        # Set the main title for the entire figure
        fig.canvas.manager.set_window_title(f"Demo: {title}")
        
        # Store all handles and labels
        all_handles = []
        all_labels = []

        if not isinstance(axs, np.ndarray):
            axs = [axs]

        for i, (data_type, y_label) in enumerate(zip(data, y_labels)):
            for filename, values in data_type:
                line, = axs[i].plot(values, label=filename)
                all_handles.append(line)
                all_labels.append(filename)
            axs[i].set_xlabel("Step Index")
            axs[i].set_ylabel(y_label)
            axs[i].set_title(y_label, fontsize=12)
            axs[i].grid(True, linestyle='--', alpha=0.7)  # Add gridlines
        
        # Remove duplicate labels while preserving order
        by_label = dict(zip(all_labels, all_handles))
        unique_labels = list(dict.fromkeys(all_labels))
        unique_handles = [by_label[label] for label in unique_labels]
        # Create a single legend for all subplots
        fig.legend(unique_handles, unique_labels)
        # Adjust layout to accommodate the legend
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        # Adjust the spacing between subplots
        fig.subplots_adjust(hspace=0.4)

    fig1, axs1 = plt.subplots(4, 1, figsize=(14, 20))
    create_plot(fig1, axs1, [position_data, velocity_data, force_data, lift_data], f'{data_input_folder} Plots',
                ["Position", "Velocity", "Force", "Lift"])
    
    # Create the third figure for force/velocity ratio
    fig3, axs3 = plt.subplots(figsize=(14, 10))
    create_plot(fig3, axs3, [f_v_ratios], f'{data_input_folder} Force/Velocity Ratio Plots',
                ["Force/Velocity Ratio"])

    fig1.savefig(f'{data_input_folder}_Variables.png', bbox_inches='tight', dpi=300)
    fig3.savefig(f'{data_input_folder}_FVratio.png', bbox_inches='tight', dpi=300)

def plot_discrete_data(descrete_data_input_folder, smoothing = False):
    # Initialize empty lists to store the data for discretised demos
    x_data = []
    v_data = []
    f_data = []
    l_data = []  
    fv_r_data = []

    for filename in os.listdir(descrete_data_input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(descrete_data_input_folder, filename)
            
            # Read the data from the CSV file
            with open(file_path, "r") as file:
                csv_reader = csv.reader(file)
                data = list(csv_reader)

            # Extract the values from the second column (index 1)
            coordinates = [eval(row[1]) for row in data if len(row) > 1]
            
            # Extract x, v, f, l values from the coordinates
            x_values = [coord[0] for coord in coordinates]
            v_values = [coord[1] for coord in coordinates]
            f_values = [coord[2] for coord in coordinates]
            l_values = [coord[3] for coord in coordinates] 
            fv_ratios = fv_ratio_calc(f_values, v_values)
    
            
            # Store the data for each file
            x_data.append((filename, x_values))
            v_data.append((filename, v_values))
            f_data.append((filename, f_values))
            l_data.append((filename, l_values))
            fv_r_data.append((filename, fv_ratios))

    def create_plot(fig, axs, data, title, y_labels):
        # Set the main title for the entire figure
        fig.canvas.manager.set_window_title(f"Demo: {title}")
        
        # Store all handles and labels
        all_handles = []
        all_labels = []

        if not isinstance(axs, np.ndarray):
            axs = [axs]

        for i, (data_type, y_label) in enumerate(zip(data, y_labels)):
            for filename, values in data_type:
                line, = axs[i].plot(values, label=filename)
                all_handles.append(line)
                all_labels.append(filename)
            axs[i].set_xlabel("Step Index")
            axs[i].set_ylabel(y_label)
            axs[i].set_title(y_label, fontsize=12)
            axs[i].grid(True, linestyle='--', alpha=0.7)  # Add gridlines
        
        # Remove duplicate labels while preserving order
        by_label = dict(zip(all_labels, all_handles))
        unique_labels = list(dict.fromkeys(all_labels))
        unique_handles = [by_label[label] for label in unique_labels]
        # Create a single legend for all subplots
        fig.legend(unique_handles, unique_labels)
        # Adjust layout to accommodate the legend
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        # Adjust the spacing between subplots
        fig.subplots_adjust(hspace=0.4)


    fig1, axs1 = plt.subplots(4, 1, figsize=(14, 20))
    create_plot(fig1, axs1, [x_data, v_data, f_data, l_data], f'{descrete_data_input_folder} Plots',
                ["Discrete Position", "Discrete Velocity", "Discrete Force", "Discrete Lift"])
    
    fig2, axs2 = plt.subplots(figsize=(14, 10))
    create_plot(fig2, axs2, [fv_r_data], f'{descrete_data_input_folder} Force/Velocity Ratio Plots',
                ["Force/Velocity Ratio"])

    fig1.savefig(f'{descrete_data_input_folder}_Variables.png', bbox_inches='tight', dpi=300)
    fig2.savefig(f'{descrete_data_input_folder}_FVratio.png', bbox_inches='tight', dpi=300)

# weird plots for each demo:
def plot_subtask_parallel_coordinates(discretised_demos, demo_indices=None):
    if demo_indices is None:
        demo_indices = range(len(discretised_demos))
    
    for idx in demo_indices:
        if idx >= len(discretised_demos):
            print(f"Warning: Demo index {idx} is out of range. Skipping.")
            continue
        
        demo = discretised_demos[idx]
        
        if not demo:
            print(f"Warning: Demo {idx} is empty. Skipping.")
            continue
        
        # Convert demo to DataFrame with explicit column names
        df = pd.DataFrame(demo, columns=['Position', 'Velocity', 'Force', 'Lift', 'Subtask'])
        
        # Normalize the data
        for col in df.columns[:-1]:  # Exclude 'Subtask' column
            if df[col].max() != df[col].min():
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            else:
                df[col] = 0  # Set to constant if all values are the same
        
        # Create a color map with distinctly different colors
        unique_subtasks = sorted(df['Subtask'].unique())
        n_colors = len(unique_subtasks)
        colors = plt.cm.tab10(np.linspace(0, 1, n_colors))  # Using tab10 for distinct colors
        color_map = dict(zip(unique_subtasks, colors))
        
        # Plot
        plt.figure(figsize=(12, 6))
        for subtask in unique_subtasks:
            subset = df[df['Subtask'] == subtask]
            if not subset.empty:
                parallel_coordinates(subset, class_column='Subtask', color=mcolors.rgb2hex(color_map[subtask]), alpha=0.5)
        
        plt.title(f'Parallel Coordinates Plot for Demo {idx}')
        plt.xlabel('Features')
        plt.ylabel('Normalized Values')
        
        # Create a custom legend
        legend_elements = [plt.Line2D([0], [0], color=mcolors.rgb2hex(color), lw=4, label=f'Subtask {subtask}')
                           for subtask, color in color_map.items()]
        plt.legend(handles=legend_elements, title='Subtasks', loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        plt.savefig(f'subtask_parallel_coordinates_demo_{idx}.png', bbox_inches='tight')
        plt.close()

    print("Parallel coordinates plots have been saved.")

# feature distributions among all demos for each subtask:
def plot_feature_distributions(discretised_demos):
    # Combine all demos into a single DataFrame
    all_data = []
    for idx, (filename, demo) in enumerate(discretised_demos.items()):
        df = pd.DataFrame(demo, columns=['Position', 'Velocity', 'Force', 'Lift', 'Subtask'])
        df['Demo'] = idx
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Print summary statistics
    print("Summary Statistics:")
    print(combined_df.describe())
    
    # Check for NaN values
    # print("\nNaN Values:")
    # print(combined_df.isna().sum())
    
    # Melt the DataFrame to long format for easier plotting
    melted_df = pd.melt(combined_df, id_vars=['Demo', 'Subtask'], 
                        value_vars=['Position', 'Velocity', 'Force', 'Lift'],
                        var_name='Feature', value_name='Value')
    
    # Remove any NaN values
    melted_df = melted_df.dropna()
    
    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle('Distribution of Features Across All Demonstrations', fontsize=16)
    
    features = ['Position', 'Velocity', 'Force', 'Lift']
    for i, feature in enumerate(features):
        ax = axes[i // 2, i % 2]
        sns.violinplot(x='Subtask', y='Value', data=melted_df[melted_df['Feature'] == feature], ax=ax)
        sns.stripplot(x='Subtask', y='Value', data=melted_df[melted_df['Feature'] == feature], 
                      ax=ax, color='black', alpha=0.1, jitter=True, size=1)
        ax.set_title(feature)
        ax.set_xlabel('Subtask')
        ax.set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig('feature_distributions_all_demos.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("Feature distribution plot has been saved.")

def plot_variable_across_trajectories(discretised_demos, variable, demo_indices=None):
    if demo_indices is None:
        demo_indices = range(len(discretised_demos))
    
    n_demos = len(demo_indices)
    cols = 3  # You can adjust this
    rows = math.ceil(n_demos / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), squeeze=False)
    fig.suptitle(f'{variable} Across Trajectories with Subtask Divisions', fontsize=16)
    
    for idx, demo_idx in enumerate(demo_indices):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        demo = list(discretised_demos.items())[demo_idx][1]
        df = pd.DataFrame(demo, columns=['Position', 'Velocity', 'Force', 'Lift', 'Subtask'])
        
        # Normalize the data
        df[variable] = (df[variable] - df[variable].min()) / (df[variable].max() - df[variable].min())
        
        # Plot the variable
        ax.plot(df.index, df[variable], label=variable, color='black', linewidth=2)
        
        # Color the background based on subtasks
        subtasks = df['Subtask'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(subtasks)))
        for subtask, color in zip(subtasks, colors):
            mask = df['Subtask'] == subtask
            ax.fill_between(df.index, 0, 1, where=mask, color=color, alpha=0.3, label=f'Subtask {subtask}')
        
        ax.set_title(f'Demo {demo_idx}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Normalized Value')
        ax.set_ylim(0, 1)
        
        # Only add legend to the first subplot
        if idx == 0:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Remove any unused subplots
    for idx in range(n_demos, rows*cols):
        fig.delaxes(axes[idx//cols, idx%cols])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{variable}_across_trajectories.png', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"{variable} across trajectories plot has been saved.")

def plot_filter_response(order, cutoff, fs):
    b, a = signal.butter(order, cutoff / (fs/2), btype='low', analog=False)
    w, h = signal.freqs(b, a)
    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.title('Butterworth filter frequency response')
    plt.xlabel('Frequency [radians / second]')
    plt.ylabel('Amplitude [dB]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(cutoff, color='green')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('filter.png', bbox_inches='tight', dpi=300)
    plt.close()
    

if __name__ == '__main__':
    n_p_bins = 200
    n_v_bins = 200
    n_f_bins = 200

    extract_data_and_remove_outliers = False
    if extract_data_and_remove_outliers:
        # Extract and save the filtered data
        # extract_and_save_filtered_data('test_10.07', 'test_extracted', 100)
        # extract_and_save_filtered_data('train_10.07', 'train_extracted', 100)
        # extract_and_save_filtered_data('fails', 'fails_extracted', 100)
        
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Data Extraction and Filtering Completed Last on {current_datetime}")
        print("------------------------------------------")

    discretised_test = data_manip(n_p_bins, n_v_bins, n_f_bins, 'test_extracted', 'test_discretised')[0]
    discretised_demos = data_manip(n_p_bins, n_v_bins, n_f_bins, 'train_extracted', 'train_discretised')[0]
    discretised_fails = data_manip(n_p_bins, n_v_bins, n_f_bins, 'fails_extracted', 'fails_discretised')[0]
    

    plot_data('train_extracted')
    plot_discrete_data('train_discretised')

    plot_feature_distributions(discretised_demos)

    variables = ['Position', 'Velocity', 'Force', 'Lift']
    for variable in variables:
        plot_variable_across_trajectories(discretised_demos, variable)  # Plot the variable across all trajectories 
    #plt.show()
