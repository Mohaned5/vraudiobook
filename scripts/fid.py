import wandb
import torch
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance

# Transform to prepare images for FID.
# We resize to 299x299 (Inception requirement) and convert to tensor.
fid_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),  # Scale values to [0, 1]
])

def download_wandb_images(entity, project, run_id, path_prefix, local_dir, timeout=29):
    """Download images from WandB for a specific path prefix and organize into folders."""
    api = wandb.Api(timeout=timeout)
    run_path = f"{entity}/{project}/{run_id}"
    run = api.run(run_path)

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
            steps.append(int(step))

    return sorted(set(steps))

def get_epochs_from_steps(steps, images_per_epoch=20):
    """Group steps into epochs based on jumps in step numbers."""
    epochs = []
    current_epoch = []

    for i, step in enumerate(steps):
        if i > 0 and step - steps[i - 1] > 1:
            if current_epoch:  # Avoid appending empty lists
                epochs.append(current_epoch)
            current_epoch = []
        current_epoch.append(step)

        if len(current_epoch) == images_per_epoch:
            epochs.append(current_epoch)
            current_epoch = []

    if current_epoch:  # Append the last epoch if not empty
        epochs.append(current_epoch)

    return epochs

def compute_fid(pred_dir, gt_dir, steps):
    """Compute FID for corresponding prediction and ground truth images, grouped by epochs."""
    epochs = get_epochs_from_steps(steps)
    print(f"Epochs: {epochs}")
    print(epochs[52])
    epoch_fid_scores = {}

    # Initialize WandB (adjust the project name if needed)
    wandb.init(project="panfusion-fid", name="FID-Per-Epoch")

    for epoch_index, epoch_steps in enumerate(tqdm(epochs, desc="Processing Epochs")):
        print(f"Processing epoch {epoch_index} with steps: {epoch_steps}")
        # Create a new FID instance for each epoch.
        fid = FrechetInceptionDistance(feature=2048, normalize=True)

        # Update the metric with ground truth images (flag: real=True)
        for step in epoch_steps:
            gt_path = os.path.join(gt_dir, f"{step}.jpg")
            if not os.path.exists(gt_path):
                print(f"Missing ground truth file for step {step}, skipping...")
                continue

            try:
                gt_image = fid_transform(Image.open(gt_path).convert("RGB"))
                # fid expects a batch dimension
                fid.update(gt_image.unsqueeze(0), real=True)
            except Exception as e:
                print(f"Error processing ground truth image for step {step}: {e}")

        # Update the metric with predicted images (flag: real=False)
        for step in epoch_steps:
            pred_path = os.path.join(pred_dir, f"{step}.jpg")
            if not os.path.exists(pred_path):
                print(f"Missing predicted file for step {step}, skipping...")
                continue

            try:
                pred_image = fid_transform(Image.open(pred_path).convert("RGB"))
                fid.update(pred_image.unsqueeze(0), real=False)
            except Exception as e:
                print(f"Error processing predicted image for step {step}: {e}")

        # Compute FID for the current epoch.
        try:
            epoch_fid = fid.compute().item()
        except Exception as e:
            print(f"Error computing FID for epoch {epoch_index}: {e}")
            epoch_fid = float('nan')

        epoch_fid_scores[epoch_index] = epoch_fid
        print(f"Epoch {epoch_index}: FID = {epoch_fid:.4f}")

        # Log the FID to WandB
        wandb.log({"epoch": epoch_index, "fid": epoch_fid})

    wandb.finish()

def get_steps_from_local_folder(folder_path):
    """Retrieve unique step numbers from filenames in a local folder."""
    steps = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            try:
                step = int(filename[:-4])  # Extract step number
                steps.append(step)
            except ValueError:
                print(f"Skipping invalid file: {filename}")
                continue

    return sorted(set(steps))

def main():
    local_dir = "wandb_images"
    os.makedirs(local_dir, exist_ok=True)

    pred_dir = os.path.join(local_dir, "pred")
    gt_dir = os.path.join(local_dir, "gt")
    
    # Optionally, you could download images from WandB by uncommenting the next lines:
    # entity = "your_entity"
    # project = "your_project"
    # run_id = "your_run_id"
    # path_prefix = "your_path_prefix"
    # download_wandb_images(entity, project, run_id, path_prefix, local_dir)

    steps = get_steps_from_local_folder(pred_dir)
    print(f"Steps: {steps}")

    print("Computing FID per epoch...")
    compute_fid(pred_dir, gt_dir, steps)

if __name__ == "__main__":
    main()
