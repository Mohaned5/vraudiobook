import wandb
import lpips
import torch
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm


def download_wandb_images(entity, project, run_id, path_prefix, local_dir, timeout=29):
    """Download images from WandB for a specific path prefix and organize into folders."""
    api = wandb.Api(timeout=timeout)
    run_path = f"{entity}/{project}/{run_id}"
    run = api.run(run_path)  # Use full run path

    pred_dir = os.path.join(local_dir, "pred")
    gt_dir = os.path.join(local_dir, "gt")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    steps = []

    for file in tqdm(run.files(), desc="Processing WandB files"):
        if file.name.startswith(path_prefix):
            local_path = os.path.join(local_dir, file.name)

            if "pano_pred" in file.name:
                step = file.name.split('_')[2]  # Extract step
                dest_path = os.path.join(pred_dir, f"{step}.jpg")
                if os.path.exists(dest_path):
                    print(f"Skipping already downloaded file: {dest_path}")
                    steps.append(int(step))
                    continue
            elif "pano_gt" in file.name:
                step = file.name.split('_')[2]  # Extract step
                dest_path = os.path.join(gt_dir, f"{step}.jpg")
                if os.path.exists(dest_path):
                    print(f"Skipping already downloaded file: {dest_path}")
                    steps.append(int(step))
                    continue
            else:
                continue

            print(f"Downloading file: {file.name}")
            file.download(replace=True, root=local_dir)
            os.rename(local_path, dest_path)
            steps.append(int(step))  # Store the step number

    return sorted(set(steps))


def main():
    entity = "mohaned-kcl"  # Replace with your WandB entity (username or team)
    project = "panfusion"  # Replace with your project name
    run_id = "vgbj5o3r"  # Your WandB run ID
    pred_prefix = "media/images/val/pano_pred"
    gt_prefix = "media/images/val/pano_gt"
    local_dir = "wandb_images"

    os.makedirs(local_dir, exist_ok=True)

    # Download prediction and ground truth images
    print("Downloading pred images...")
    download_wandb_images(entity, project, run_id, pred_prefix, local_dir, timeout=29)
    print("Downloading gt images...")
    download_wandb_images(entity, project, run_id, gt_prefix, local_dir, timeout=29)


if __name__ == "__main__":
    main()
