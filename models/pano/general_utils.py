# Copyright (C) 2024 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.

from typing import List, Dict
import torch

import sys 
sys.path.append(".")
sys.path.append("..")

import numpy as np
from diffusers.utils.torch_utils import randn_tensor
import torch.nn.functional as F
from skimage import filters

from tqdm import tqdm

## Attention Utils

def get_dynamic_threshold(tensor):
    """
    Compute a dynamic threshold using Otsu's method on the provided tensor.
    
    Args:
        tensor: A torch.Tensor.
    
    Returns:
        A threshold value computed from the tensor.
    """
    return filters.threshold_otsu(tensor.cpu().numpy())

def attn_map_to_binary(attention_map, scaler=1.):
    """
    Convert an attention map to a binary mask using Otsu thresholding.
    
    Args:
        attention_map: A torch.Tensor representing the attention map.
        scaler: A factor to adjust the threshold.
    
    Returns:
        A binary mask as a NumPy array.
    """
    attention_map_np = attention_map.cpu().numpy()
    threshold_value = filters.threshold_otsu(attention_map_np) * scaler
    binary_mask = (attention_map_np > threshold_value).astype(np.uint8)
    return binary_mask


## Features

def gaussian_smooth(input_tensor, kernel_size=3, sigma=1):
    """
    Apply Gaussian smoothing on each 2D slice of a 3D tensor.
    
    Args:
        input_tensor: A torch.Tensor with shape (N, H, W) (or (N, C, H, W) if multi‐channel).
        kernel_size: The size of the Gaussian kernel.
        sigma: The standard deviation of the Gaussian.
    
    Returns:
        A torch.Tensor of the same shape as input_tensor with each slice smoothed.
    """
    # Create a Gaussian kernel
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * 
                      np.exp(-((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (kernel_size, kernel_size)
    )
    kernel = torch.tensor(kernel / kernel.sum(), dtype=input_tensor.dtype, device=input_tensor.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, kernel_size, kernel_size)

    smoothed_slices = []
    for i in range(input_tensor.size(0)):
        # Handle both single-channel (2D) and multi-channel (3D) slices:
        slice_tensor = input_tensor[i] if input_tensor.dim() == 3 else input_tensor[i, :, :]
        if slice_tensor.dim() == 2:
            slice_tensor = slice_tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            smoothed = F.conv2d(slice_tensor, kernel, padding=kernel_size // 2)[0, 0]
        else:
            # For multi-channel, apply the same kernel to each channel
            smoothed = F.conv2d(slice_tensor.unsqueeze(0), kernel, padding=kernel_size // 2)[0]
        smoothed_slices.append(smoothed)
    smoothed_tensor = torch.stack(smoothed_slices, dim=0)
    return smoothed_tensor


## Dense Correspondence Utils

def cos_dist(a, b):
    """
    Compute the cosine distance between two sets of vectors.
    
    Args:
        a: A torch.Tensor of shape (N, D).
        b: A torch.Tensor of shape (M, D).
    
    Returns:
        A torch.Tensor of shape (N, M) representing the cosine distance.
    """
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)
    res = a_norm @ b_norm.T
    return 1 - res

def gen_nn_map(src_features, src_mask, tgt_features, tgt_mask, device, batch_size=100, tgt_size=64):
    """
    Generate a nearest-neighbor map from source features to target features using cosine distance.
    
    Args:
        src_features: Source feature tensor.
        src_mask: Binary mask for source features.
        tgt_features: Target feature tensor.
        tgt_mask: Binary mask for target features (not used in current implementation).
        device: Torch device.
        batch_size: Batch size for processing patches.
        tgt_size: The target spatial size (default 64 for SD2.0 latent space).
    
    Returns:
        A tuple (nearest_neighbor_indices, nearest_neighbor_distances).
    """
    # Resize features to the target latent space size
    resized_src_features = F.interpolate(src_features.unsqueeze(0), size=tgt_size, mode='bilinear', align_corners=False).squeeze(0)
    resized_src_features = resized_src_features.permute(1, 2, 0).view(tgt_size**2, -1)
    resized_tgt_features = F.interpolate(tgt_features.unsqueeze(0), size=tgt_size, mode='bilinear', align_corners=False).squeeze(0)
    resized_tgt_features = resized_tgt_features.permute(1, 2, 0).view(tgt_size**2, -1)

    nearest_neighbor_indices = torch.zeros(tgt_size**2, dtype=torch.long, device=device)
    nearest_neighbor_distances = torch.zeros(tgt_size**2, dtype=src_features.dtype, device=device)

    for i in range(0, tgt_size**2, batch_size):
        distances = cos_dist(resized_src_features, resized_tgt_features[i:i+batch_size])
        distances[~src_mask] = 2.0  # Set a high distance where the mask is False
        min_distances, min_indices = torch.min(distances, dim=0)
        nearest_neighbor_indices[i:i+batch_size] = min_indices
        nearest_neighbor_distances[i:i+batch_size] = min_distances

    return nearest_neighbor_indices, nearest_neighbor_distances

def cyclic_nn_map(features, masks, latent_resolutions, device):
    """
    Compute cyclic nearest-neighbor maps between all pairs of images in a batch.
    
    Args:
        features: A tensor of features with shape (batch, C, H, W).
        masks: A dictionary mapping each latent resolution to a binary mask tensor.
        latent_resolutions: A list of latent resolutions (e.g. [32, 64]) to compute the maps.
        device: Torch device.
    
    Returns:
        A tuple (nn_map_dict, nn_distances_dict) where each is a dictionary keyed by latent resolution.
    """
    bsz = features.shape[0]
    nn_map_dict = {}
    nn_distances_dict = {}

    for tgt_size in latent_resolutions:
        nn_map = torch.empty(bsz, bsz, tgt_size**2, dtype=torch.long, device=device)
        nn_distances = torch.full((bsz, bsz, tgt_size**2), float('inf'), dtype=features.dtype, device=device)

        for i in range(bsz):
            for j in range(bsz):
                if i != j:
                    nn_idx, nn_dist = gen_nn_map(features[j], masks[tgt_size][j], features[i], masks[tgt_size][i], device, batch_size=None, tgt_size=tgt_size)
                    nn_map[i, j] = nn_idx
                    nn_distances[i, j] = nn_dist

        nn_map_dict[tgt_size] = nn_map
        nn_distances_dict[tgt_size] = nn_distances

    return nn_map_dict, nn_distances_dict

def anchor_nn_map(features, anchor_features, masks, anchor_masks, latent_resolutions, device):
    """
    Compute nearest-neighbor mappings between feature sets and anchor feature sets.
    
    Args:
        features: Feature tensor for the main images.
        anchor_features: Feature tensor for the anchor images.
        masks: Dictionary of masks for the main images.
        anchor_masks: Dictionary of masks for the anchor images.
        latent_resolutions: A list of latent resolutions (e.g. [32, 64]).
        device: Torch device.
    
    Returns:
        A tuple (nn_map_dict, nn_distances_dict) mapping each main image to its anchor features.
    """
    bsz = features.shape[0]
    anchor_bsz = anchor_features.shape[0]
    nn_map_dict = {}
    nn_distances_dict = {}

    for tgt_size in latent_resolutions:
        nn_map = torch.empty(bsz, anchor_bsz, tgt_size**2, dtype=torch.long, device=device)
        nn_distances = torch.full((bsz, anchor_bsz, tgt_size**2), float('inf'), dtype=features.dtype, device=device)

        for i in range(bsz):
            for j in range(anchor_bsz):
                nn_idx, nn_dist = gen_nn_map(anchor_features[j], anchor_masks[tgt_size][j], features[i], masks[tgt_size][i], device, batch_size=None, tgt_size=tgt_size)
                nn_map[i, j] = nn_idx
                nn_distances[i, j] = nn_dist

        nn_map_dict[tgt_size] = nn_map
        nn_distances_dict[tgt_size] = nn_distances

    return nn_map_dict, nn_distances_dict
