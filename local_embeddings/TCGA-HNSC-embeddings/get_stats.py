import os
import torch
import numpy as np
from tqdm import tqdm

def get_pt_file_dimensions(file_path):
    tensor = torch.load(file_path, map_location=torch.device('cpu'))
    return tensor.shape

# Initialize lists to store dimensions
dimensions = []
file_names = []

# Iterate over files in the current directory
for file_name in tqdm(os.listdir('.')):
    if file_name.endswith('.pt'):
        dimensions.append(get_pt_file_dimensions(file_name)[0])
        file_names.append(file_name)

# Prepare the output string
output = []

# Print and store file names and their dimensions
for file_name, dim in zip(file_names, dimensions):
    info = f"File: {file_name}, Dimensions: {dim}x192"
    print(info)
    output.append(info)

# Calculate and print statistical values for n
if dimensions:
    max_n = max(dimensions)
    min_n = min(dimensions)
    avg_n = np.mean(dimensions)
    median_n = np.median(dimensions)

    stats_info = (
        f"\nTotal .pt files: {len(file_names)}\n"
        f"Maximal n: {max_n}\n"
        f"Minimal n: {min_n}\n"
        f"Average n: {avg_n:.2f}\n"
        f"Median n: {median_n}"
    )
    print(stats_info)
    output.append(stats_info)
else:
    no_files_info = "\nNo .pt files found in the current directory."
    print(no_files_info)
    output.append(no_files_info)

# Write the output to a text file
with open('pt_files_info.txt', 'w') as f:
    for line in output:
        f.write(line + '\n')
