#!/bin/bash

# Destination folder
destination="clinical"

# Create the destination folder if it doesn't exist
mkdir -p "$destination"

# Loop through clinical0 to clinical9
for i in {0..9}; do
    folder="clinical$i"
    
    # Check if the folder exists
    if [ -d "$folder" ]; then
        # Move all contents from the folder to the destination
        mv "$folder"/* "$destination"/
        
        # Optionally, remove the empty folder after moving its contents
        rmdir "$folder"
    else
        echo "Folder $folder does not exist, skipping..."
    fi
done

echo "All contents moved to $destination/"