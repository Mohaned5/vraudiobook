import wandb
import lpips
import torch
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

# Initialize the LPIPS model
loss_fn = lpips.LPIPS(net='alex')  # Options: 'alex', 'vgg', 'squeeze'

# Transform to prepare images for LPIPS
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

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

def compute_lpips(pred_dir, gt_dir, steps):
    """Compute LPIPS score for corresponding prediction and ground truth images, grouped by epochs."""
    epochs = get_epochs_from_steps(steps)
    print(f"Epochs: {epochs}")
    epoch_lpips_scores = {}

    for epoch_index, epoch_steps in enumerate(tqdm(epochs, desc="Processing Epochs")):
        print(f"Processing epoch {epoch_index} with steps: {epoch_steps}")
        lpips_scores = []

        for step in epoch_steps:
            pred_path = os.path.join(pred_dir, f"{step}.jpg")
            gt_path = os.path.join(gt_dir, f"{step}.jpg")

            if not os.path.exists(pred_path) or not os.path.exists(gt_path):
                print(f"Missing files for step {step}, skipping...")
                continue

            try:
                pred_image = transform(Image.open(pred_path).convert("RGB")).unsqueeze(0)
                gt_image = transform(Image.open(gt_path).convert("RGB")).unsqueeze(0)

                # Compute LPIPS score
                score = loss_fn(pred_image, gt_image)
                lpips_scores.append(score.item())
            except Exception as e:
                print(f"Error processing step {step}: {e}")

        if lpips_scores:
            avg_lpips = sum(lpips_scores) / len(lpips_scores)
        else:
            avg_lpips = float('nan')

        epoch_lpips_scores[epoch_index] = avg_lpips

        print(f"Epoch {epoch_index}: Average LPIPS = {avg_lpips:.4f}")

    # Log results to WandB
    wandb.init(project="panfusion-lpips", name="LPIPS-Per-Epoch")
    for epoch, avg_lpips in epoch_lpips_scores.items():
        wandb.log({"epoch": epoch, "average_lpips": avg_lpips})
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
    steps = get_steps_from_local_folder("wandb_images/pred")
    print(f"Steps: {steps}")

    # Compute LPIPS for each step
    print("Computing LPIPS...")
    compute_lpips(pred_dir, gt_dir, steps)


if __name__ == "__main__":
    main()
