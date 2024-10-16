import os
from data_manip import extract_and_save_filtered_data, plot_data

def process_demo_folders(base_folder):
    # Get all subdirectories in the base folder
    demo_folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]

    for demo_folder in demo_folders:
        input_folder = os.path.join(base_folder, demo_folder)
        extracted_folder = f"{demo_folder}_extracted"
        
        print(f"Processing demonstrations in folder: {demo_folder}")
        
        # Extract and save filtered data
        extract_and_save_filtered_data(input_folder, extracted_folder, sampling_frequency=100)
        
        # Plot the extracted data
        plot_data(extracted_folder)
        
        print(f"Completed processing for {demo_folder}")
        print("-------------------------------------------")

if __name__ == "__main__":
    base_folder = "new_demonstrations"  # Update this to the path of your folder containing the demonstration folders
    process_demo_folders(base_folder)