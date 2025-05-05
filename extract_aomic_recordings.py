import os
import stat
import tarfile
import numpy as np
import shutil
from pathlib import Path

from bale import PROJECT_PATH

input_folder =  Path(f"{PROJECT_PATH.__str__()}/ds002785")
temp_folder = Path(f"{PROJECT_PATH.__str__()}/recordings")
final_output_folder = Path(f"{PROJECT_PATH.__str__()}/data/aomic/recordings")

# Create needed folders
os.makedirs(temp_folder, exist_ok=True)
os.makedirs(final_output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".tar") and "task-faces" in file:
        tar_path = os.path.join(input_folder, file)
        print(f"Extracting: {file}")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=temp_folder)

converted_files = []

for root, dirs, files in os.walk(temp_folder):
    for file in files:
        if file.endswith("bold.pyd"):
            full_path = os.path.join(root, file)
            npy_path = full_path.replace(".bold.pyd", ".npy")

            try:
                with open(full_path, "rb") as f:
                    data = np.load(f, allow_pickle=True)

                np.save(npy_path, data)
                print(f"Converted: {full_path} -> {npy_path}")
                converted_files.append(npy_path)
            except Exception as e:
                print(f"Failed to convert {full_path}: {e}")

for npy_file in converted_files:
    filename = os.path.basename(npy_file)
    dest_path = os.path.join(final_output_folder, filename)
    shutil.copy2(npy_file, dest_path)
    print(f"Copied: {npy_file} -> {dest_path}")

def force_remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

try:
    shutil.rmtree(temp_folder, onerror=force_remove_readonly)
    print(f"Removed temporary folder: {temp_folder}")
except Exception as e:
    print(f"Could not delete {temp_folder}: {e}")
