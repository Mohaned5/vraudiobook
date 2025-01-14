import numpy as np
from PIL import Image
import os
import json
from Perspective_and_Equirectangular import e2p
import torch

# Function to generate perspective images from a panorama
def split_panorama_to_perspectives(panorama_path, output_folder):
    # Load the panorama
    panorama = Image.open(panorama_path)
    pano_width, pano_height = panorama.size

    # Convert panorama to tensor
    panorama_tensor = torch.tensor(np.array(panorama).transpose(2, 0, 1)).unsqueeze(0) / 255.0  # Shape (1, 3, H, W)

    # Parameters
    FoV = 90  # Field of View in degrees
    pers_h, pers_w = 512, 512  # Perspective image size

    # Define camera positions (PanFusion requires 20 views)
    theta = torch.tensor([0, 72, 144, 216, 288, 36, 108, 180, 252, 324, 0, 72, 144, 216, 288, 36, 108, 180, 252, 324])
    phi = torch.tensor([0, 0, 0, 0, 0, 45, 45, 45, 45, 45, -45, -45, -45, -45, -45, 90, 90, -90, -90, -90])
    B, M = 1, len(theta)

    cameras = {
        'FoV': torch.full((B, M), FoV),
        'theta': theta.unsqueeze(0).repeat(B, 1),
        'phi': phi.unsqueeze(0).repeat(B, 1)
    }

    # Generate perspective views
    perspective_views = generate_perspective_views(panorama_tensor, cameras, pers_h, pers_w)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save perspective views
    for i in range(M):
        perspective_image = (perspective_views[0, i] * 255).byte().permute(1, 2, 0).cpu().numpy()
        perspective_pil = Image.fromarray(perspective_image)
        perspective_pil.save(os.path.join(output_folder, f"{i}.png"))

    return {
        'FoV': cameras['FoV'].tolist(),
        'theta': cameras['theta'].tolist(),
        'phi': cameras['phi'].tolist()
    }

# Function to project panorama to perspective views
def generate_perspective_views(panorama_tensor, cameras, pers_h=512, pers_w=512):
    B, C, H, W = panorama_tensor.shape
    M = cameras['FoV'].shape[1]

    perspective_views = []

    for b in range(B):
        batch_views = []
        for m in range(M):
            fov = cameras['FoV'][b, m].item()
            theta = cameras['theta'][b, m].item()
            phi = cameras['phi'][b, m].item()

            view = e2p(
                panorama_tensor[b:b+1],
                fov,
                theta,
                phi,
                (pers_h, pers_w),
                mode='bilinear'  # Use bilinear interpolation for smooth lines
            )
            batch_views.append(view)
        perspective_views.append(torch.cat(batch_views, dim=0))

    return torch.stack(perspective_views, dim=0)

# Function to process dataset and generate perspective views
def process_dataset(dataset_path, output_root):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    output_train_folder = os.path.join(output_root, 'val')
    os.makedirs(output_train_folder, exist_ok=True)

    new_dataset_path = os.path.join(output_root, 'val2.json')

    # Initialize a new dataset file
    if not os.path.exists(new_dataset_path):
        with open(new_dataset_path, 'w') as f:
            json.dump([], f, indent=4)

    for i, entry in enumerate(dataset):
        # Extract image path and prepare output directories
        print(f"Processing entry {i+1}/{len(dataset)}")
        pano_image_path = entry['pano']
        prompt = entry['pano_prompt'].replace(' ', '_')  # Replace spaces with underscores for folder names
        entry_folder = os.path.join(output_train_folder, prompt)
        pano_folder = os.path.join(entry_folder, 'pano')
        images_folder = os.path.join(entry_folder, 'images')
        os.makedirs(pano_folder, exist_ok=True)
        os.makedirs(images_folder, exist_ok=True)

        # Copy panorama to pano folder
        pano_output_path = os.path.join(pano_folder, os.path.basename(pano_image_path))
        Image.open(pano_image_path).save(pano_output_path)

        # Generate perspective views
        cameras = split_panorama_to_perspectives(pano_image_path, images_folder)

        # Update entry with new keys
        new_entry = {
            'pano': f"val/{prompt}/pano/{os.path.basename(pano_image_path)}",
            'pano_prompt': entry['pano_prompt'],
            'images': [f"val/{prompt}/images/{j}.png" for j in range(20)],
            'cameras': cameras
        }

        # Append new entry to dataset file incrementally
        with open(new_dataset_path, 'r+') as f:
            current_data = json.load(f)
            current_data.append(new_entry)
            f.seek(0)
            json.dump(current_data, f, indent=4)

# Example Usage
process_dataset('val.json', 'output')
