import torch
import os
from tqdm import tqdm  # For progress bars (install with `pip install tqdm`)

def reset_pt_files_to_cpu(directory):
    """
    Safely resets all .pt files in `directory` to CPU-only format.
    Original files are backed up with a `.bak` extension.
    """
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('.pt'):
            filepath = os.path.join(directory, filename)
            
            # Skip if backup already exists (avoid reprocessing)
            backup_path = filepath + '.bak'
            if os.path.exists(backup_path):
                print(f"Skipping {filename} (backup already exists)")
                continue
            
            try:
                # Load tensor (force CPU to strip GPU info)
                tensor = torch.load(filepath, map_location='cpu')
                
                # Backup original file
                os.rename(filepath, backup_path)
                
                # Save as CPU-only
                torch.save(tensor, filepath)
                print(f"Reset {filename} to CPU-only (backup: {filename}.bak)")
                
            except Exception as e:
                print(f"ERROR processing {filename}: {str(e)}")
                # Restore backup if something went wrong
                if os.path.exists(backup_path):
                    os.rename(backup_path, filepath)

# Usage:
reset_pt_files_to_cpu('TCGA-UCEC-embeddings')