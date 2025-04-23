import os
import numpy as np
import pandas as pd

# Define the base directory containing the batch folders
data_dir = '../data/'
base_dir = data_dir + 'features/w512_o32_max_diff_first'

# List of all batch folders
batch_folders = [folder for folder in os.listdir(base_dir) if folder.startswith('batch_')]

# Initialize lists to collect data
all_diff_features = []
all_image_ids = []
all_labels = []

# Iterate over each batch folder
for folder in batch_folders:
    folder_path = os.path.join(base_dir, folder)
    
    # Load diff_features.npy
    diff_features_path = os.path.join(folder_path, 'diff_features.npy')
    diff_features = np.load(diff_features_path)
    
    # Load image_ids.txt
    image_ids_path = os.path.join(folder_path, 'image_ids.txt')
    with open(image_ids_path, 'r') as file:
        image_ids = file.read().splitlines()
            
    # Load labels.npy
    labels_path = os.path.join(folder_path, 'labels.npy')
    labels = np.load(labels_path)

    # make sure that they are the same length
    if not len(diff_features)==len(image_ids)==len(labels):
        print(f"files in {folder} are not of same length")
        break
    
    all_diff_features.append(diff_features)
    all_image_ids.extend(image_ids)
    all_labels.append(labels)

# Concatenate all diff_features arrays into a single array
all_diff_features = np.concatenate(all_diff_features, axis=0)

# Confirm that image_ids and labels have the same length
if len(all_image_ids) != sum(len(lbl) for lbl in all_labels):
    raise ValueError("The total number of image ids does not match the total number of labels.")

# Concatenate all labels into a single array
all_labels = np.concatenate(all_labels, axis=0)

# Create a DataFrame for the CSV file
data = {
    'timeline_id': all_image_ids,
    'event': all_labels
}
df = pd.DataFrame(data)

# Save the concatenated results
np.save(os.path.join(data_dir, 'diff_features.npy'), all_diff_features)
df.to_csv(os.path.join(data_dir, 'feature_annotations.csv'), index=False)

print("Data concatenation complete. Files saved as 'concatenated_diff_features.npy' and 'concatenated_data.csv'.")
