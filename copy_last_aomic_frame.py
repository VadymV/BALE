# Define the source and destination folders
import os
import re
import shutil
from pathlib import Path

from bale import PROJECT_PATH

source_folder = Path(f"{PROJECT_PATH.__str__()}/frames")
destination_folder = Path(f"{PROJECT_PATH.__str__()}/data/aomic/images")

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Function to find the image with the largest integer in its filename
def get_largest_image(folder_path):
    largest_image = None
    largest_int = -1

    for filename in os.listdir(folder_path):
        # Match the 'frame_xxxx.jpg' pattern
        match = re.match(r'frame_(\d+)\.jpg', filename)
        if match:
            current_int = int(match.group(1))
            if current_int > largest_int:
                largest_int = current_int
                largest_image = filename

    return largest_image


# Walk through the 'frames' folder
for region in ['NorthEuropean', 'Mediterranean']:
    region_folder = os.path.join(source_folder, region)

    for subfolder in os.listdir(region_folder):
        subfolder_path = os.path.join(region_folder, subfolder)

        if os.path.isdir(subfolder_path):
            # Get the largest image from the current subfolder
            largest_image_name = get_largest_image(subfolder_path)

            if largest_image_name:
                # Build the new filename
                new_filename = f"{region}_{subfolder}_{largest_image_name}"
                # Define the source image path
                source_image_path = os.path.join(subfolder_path,
                                                 largest_image_name)
                # Define the destination image path
                destination_image_path = os.path.join(destination_folder,
                                                      new_filename)

                # Copy the largest image to the destination folder
                shutil.copy(source_image_path, destination_image_path)
            else:
                raise ValueError(f"No image found in {subfolder_path}")

print("Images copied successfully.")