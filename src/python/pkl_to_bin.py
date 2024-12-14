import numpy as np
import os
import pickle
import struct
import open3d as o3d

# ChatGPT was used to create this script

# Path to the data folder
data_folder = "Data"

# Camera intrinsics
camera_intrinsics = np.array(
    [[975.482117, 0.0, 1019.53790], [0.0, 975.301147, 776.480408], [0, 0, 1]]
)

# Iterate through all files in the folder
for file_name in os.listdir(data_folder):
    if file_name.endswith(".pkl"):
        pkl_file_path = os.path.join(data_folder, file_name)

        with open(pkl_file_path, "rb") as pkl_file:
            try:
                depth_data = pickle.load(pkl_file)
            except Exception as e:
                print(f"Failed to load {pkl_file_path}: {e}")
                continue

        # Verify the depth data dimensions
        if len(depth_data.shape) != 2:
            print(f"Invalid depth data shape in {file_name}. Expected 2D array.")
            continue

        # Generate corresponding image path
        image_file_path = os.path.join(data_folder, file_name.replace(".pkl", ".png"))
        if not os.path.exists(image_file_path):
            print(f"Image file corresponding to {file_name} not found.")
            continue

        # Load the image
        rgb_image = o3d.io.read_image(image_file_path)
        if rgb_image is None:
            print(f"Failed to load image: {image_file_path}")
            continue

        rgb_image = np.asarray(rgb_image)
        height, width = depth_data.shape
        fx, fy, cx, cy = (
            camera_intrinsics[0, 0],
            camera_intrinsics[1, 1],
            camera_intrinsics[0, 2],
            camera_intrinsics[1, 2],
        )

        # Create 3D points manually
        points = []
        for v in range(height):
            for u in range(width):
                z = depth_data[v, u]
                if z > 0:  # Filter out invalid depths
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    r, g, b = rgb_image[v, u]
                    points.append((x, y, z, r, g, b))

        # Save to binary format
        bin_file_name = file_name.replace(".pkl", ".bin")
        bin_file_path = os.path.join(data_folder, bin_file_name)
        try:
            with open(bin_file_path, "wb") as bin_file:
                for point in points:
                    bin_file.write(
                        struct.pack("fffBBB", *point)
                    )  # fff for x, y, z; BBB for r, g, b
            print(f"Converted {file_name} to {bin_file_name}")
        except Exception as e:
            print(f"Failed to save {bin_file_name}: {e}")
