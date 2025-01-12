import numpy as np
from PIL import Image
import os
import json
from Perspective_and_Equirectangular import e2p
import torch
from einops import rearrange

# Function to generate perspective images from a panorama
def split_panorama_to_perspectives(panorama_path, output_folder):
    # Load the panorama
    panorama = Image.open(panorama_path)
    pano_width, pano_height = panorama.size

    # Convert panorama to tensor
    panorama_tensor = torch.tensor(np.array(panorama).transpose(2, 0, 1)).unsqueeze(0) / 255.0  # Shape (1, 3, H, W)

    # Parameters
    FoV = 90  # Field of View in degrees
    pers_h, pers_w = 256, 256  # Perspective image size for each slice

    # Define camera positions for 8 views (each shifted by 45 degrees)
    theta = torch.tensor([0, 45, 90, 135, 180, 225, 270, 315], dtype=torch.float32)
    phi = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
    num_views = len(theta)
    B = 1

    cameras = {
        'FoV': torch.full((B, num_views), FoV, dtype=torch.float32),
        'theta': theta.unsqueeze(0).repeat(B, 1),
        'phi': phi.unsqueeze(0).repeat(B, 1)
    }

    # Generate perspective views
    perspective_views = generate_perspective_views(panorama_tensor, cameras, pers_h, pers_w)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save perspective views
    for i in range(num_views):
        perspective_image = (perspective_views[0, i] * 255).byte().permute(1, 2, 0).cpu().numpy()
        perspective_pil = Image.fromarray(perspective_image)
        perspective_pil.save(os.path.join(output_folder, f"{i}.png"))

    return {
        'FoV': cameras['FoV'].tolist(),
        'theta': cameras['theta'].tolist(),
        'phi': cameras['phi'].tolist()
    }

# Function to project panorama to perspective views
def generate_perspective_views(panorama_tensor, cameras, pers_h=256, pers_w=256):
    B, C, H, W = panorama_tensor.shape
    num_views = cameras['FoV'].shape[1]

    perspective_views = []

    for b in range(B):
        batch_views = []
        for m in range(num_views):
            fov = cameras['FoV'][b, m].item()
            theta = cameras['theta'][b, m].item()
            phi = cameras['phi'][b, m].item()

            view = e2p(
                panorama_tensor[b:b+1],
                fov,
                theta,
                phi,
                (pers_h, pers_w),
                mode='nearest'
            )
            batch_views.append(view)
        perspective_views.append(torch.cat(batch_views, dim=0))
    
    return torch.stack(perspective_views, dim=0)

# Function to process dataset and generate perspective views
def process_dataset(dataset_path, output_root):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    output_train_folder = os.path.join(output_root, 'train')
    os.makedirs(output_train_folder, exist_ok=True)

    new_dataset_path = os.path.join(output_root, 'train.json')

    # Initialize a new dataset file if it doesn't exist
    if not os.path.exists(new_dataset_path):
        with open(new_dataset_path, 'w') as f:
            json.dump([], f, indent=4)

    for i, entry in enumerate(dataset):
        # Extract image path
        pano_image_path = entry['pano']

        # Skip if panorama file doesn't exist
        if not os.path.exists(pano_image_path):
            print(f"Skipping entry {i+1}/{len(dataset)}: File not found - {pano_image_path}")
            continue

        print(f"Processing entry {i+1}/{len(dataset)}")
        entry_name = os.path.splitext(os.path.basename(pano_image_path))[0]
        entry_folder = os.path.join(output_train_folder, entry_name)
        pano_folder = os.path.join(entry_folder, 'pano')
        images_folder = os.path.join(entry_folder, 'images')
        os.makedirs(pano_folder, exist_ok=True)
        os.makedirs(images_folder, exist_ok=True)

        # Copy panorama to pano folder
        pano_output_path = os.path.join(pano_folder, os.path.basename(pano_image_path))
        Image.open(pano_image_path).save(pano_output_path)

        # Generate perspective views and get updated camera data (8 images now)
        cameras = split_panorama_to_perspectives(pano_image_path, images_folder)

        # Update the images list paths (now 8 images: 0.png to 7.png)
        images_paths = [f"train/{entry_name}/images/{j}.png" for j in range(8)]

        new_entry = {
            'pano': f"train/{entry_name}/pano/{os.path.basename(pano_image_path)}",
            'pano_prompt': entry.pop('pano_prompt') if 'pano_prompt' in entry else "",
            'images': images_paths,
            'cameras': cameras,
            'mood': entry.get('mood', ''),
            'tags': entry.get('tags', []),
            'lighting': entry.get('lighting', ''),
            'split': entry.get('split', ''),
            'negative_tags': entry.get('negative_tags', []),
            'prompts': []  # Leaving prompts empty as requested
        }

        # Append new entry to dataset file incrementally
        with open(new_dataset_path, 'r+') as f:
            current_data = json.load(f)
            current_data.append(new_entry)
            f.seek(0)
            json.dump(current_data, f, indent=4)

# Example Usage
if __name__ == "__main__":
    process_dataset('train.json', 'output')
