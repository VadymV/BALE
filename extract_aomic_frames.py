import os
import subprocess

# Define input and output directories
input_dirs = {
    "Mpeg_videos_Faceforward_Mediterranean_ZIPfile_approx_400Mb": "Mediterranean",
    "Mpeg_videos_Faceforward_NorthEuropean_ZIPfile_approx_550Mb": "NorthEuropean"
}

output_root = "frames"
os.makedirs(output_root, exist_ok=True)

# Loop through both input folders
for input_dir, region in input_dirs.items():
    region_output_dir = os.path.join(output_root, region)
    os.makedirs(region_output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".mpeg"):
            video_path = os.path.join(input_dir, filename)
            video_name = os.path.splitext(filename)[0]

            video_output_dir = os.path.join(region_output_dir, video_name)
            os.makedirs(video_output_dir, exist_ok=True)

            output_pattern = os.path.join(video_output_dir, "frame_%04d.jpg")
            command = [
                "ffmpeg",
                "-i", video_path,
                output_pattern
            ]

            print(f"Extracting frames from: {video_path}")
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

print("Frame extraction complete.")
