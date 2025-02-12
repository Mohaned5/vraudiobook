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

##############################
#      Attention Utils       #
##############################

def get_dynamic_threshold(tensor: torch.Tensor) -> float:
    """
    Computes a dynamic threshold using Otsu's method.
    
    Args:
        tensor: A torch.Tensor.
    
    Returns:
        A threshold value computed from the tensor.
    """
    return filters.threshold_otsu(tensor.cpu().numpy())

def attn_map_to_binary(attention_map: torch.Tensor, scaler: float = 1.) -> np.ndarray:
    """
    Converts an attention map into a binary mask using Otsu thresholding.
    
    Args:
        attention_map: A torch.Tensor representing the attention map.
        scaler: A multiplicative factor for the threshold.
    
    Returns:
        A NumPy binary mask.
    """
    attention_map_np = attention_map.cpu().numpy()
    threshold_value = filters.threshold_otsu(attention_map_np) * scaler
    binary_mask = (attention_map_np > threshold_value).astype(np.uint8)
    return binary_mask

##############################
#         Features           #
##############################

def gaussian_smooth(input_tensor: torch.Tensor, kernel_size: int = 3, sigma: float = 1) -> torch.Tensor:
    """
    Applies Gaussian smoothing on each 2D slice of an input tensor.
    
    Args:
        input_tensor: A tensor of shape (N, H, W) (or (N, C, H, W) for multi–channel).
        kernel_size: The size of the Gaussian kernel.
        sigma: The standard deviation for the Gaussian.
    
    Returns:
        A tensor with the same shape as input_tensor, with each 2D slice smoothed.
    """
    # Create a Gaussian kernel
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * 
                      np.exp(-(((x - (kernel_size - 1) / 2) ** 2) + ((y - (kernel_size - 1) / 2) ** 2)) / (2 * sigma ** 2)),
        (kernel_size, kernel_size)
    )
    kernel = torch.tensor(kernel / kernel.sum(), dtype=input_tensor.dtype, device=input_tensor.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, kernel_size, kernel_size)

    smoothed_slices = []
    # Process each slice independently.
    for i in range(input_tensor.size(0)):
        # Handle both single-channel (2D) and multi-channel slices.
        slice_tensor = input_tensor[i]
        if slice_tensor.dim() == 2:
            slice_tensor = slice_tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            smoothed = F.conv2d(slice_tensor, kernel, padding=kernel_size // 2)[0, 0]
        else:
            # Assume shape (C, H, W): apply same kernel to each channel
            smoothed = F.conv2d(slice_tensor.unsqueeze(0), kernel, padding=kernel_size // 2)[0]
        smoothed_slices.append(smoothed)
    smoothed_tensor = torch.stack(smoothed_slices, dim=0)
    return smoothed_tensor

##############################
#   Dense Correspondence     #
##############################

def cos_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine distance between each pair of vectors from a and b.
    
    Args:
        a: Tensor of shape (N, D)
        b: Tensor of shape (M, D)
    
    Returns:
        A tensor of shape (N, M) with cosine distances.
    """
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)
    res = a_norm @ b_norm.T
    return 1 - res

def gen_nn_map(src_features: torch.Tensor, src_mask: torch.Tensor,
               tgt_features: torch.Tensor, tgt_mask: torch.Tensor,
               device: torch.device, batch_size: int = 100, tgt_size: int = 64) -> (torch.Tensor, torch.Tensor):
    """
    Generates a nearest–neighbor map between source and target features using cosine distance.
    (For SD2.0, the latent spatial resolution is typically 64×64.)
    
    Args:
        src_features: Source features tensor.
        src_mask: Boolean mask for the source.
        tgt_features: Target features tensor.
        tgt_mask: Boolean mask for the target (unused in this version).
        device: Torch device.
        batch_size: Batch size for processing patches.
        tgt_size: The target size for spatial dimensions (default is 64).
    
    Returns:
        A tuple (nearest_neighbor_indices, nearest_neighbor_distances), each of shape (tgt_size**2,).
    """
    # Resize both source and target features to (tgt_size, tgt_size)
    resized_src_features = F.interpolate(src_features.unsqueeze(0), size=tgt_size, mode='bilinear', align_corners=False).squeeze(0)
    resized_src_features = resized_src_features.permute(1, 2, 0).view(tgt_size ** 2, -1)
    resized_tgt_features = F.interpolate(tgt_features.unsqueeze(0), size=tgt_size, mode='bilinear', align_corners=False).squeeze(0)
    resized_tgt_features = resized_tgt_features.permute(1, 2, 0).view(tgt_size ** 2, -1)

    nearest_neighbor_indices = torch.zeros(tgt_size ** 2, dtype=torch.long, device=device)
    nearest_neighbor_distances = torch.zeros(tgt_size ** 2, dtype=src_features.dtype, device=device)

    for i in range(0, tgt_size ** 2, batch_size):
        distances = cos_dist(resized_src_features, resized_tgt_features[i:i + batch_size])
        distances[~src_mask] = 2.0  # assign high distance where mask is False
        min_distances, min_indices = torch.min(distances, dim=0)
        nearest_neighbor_indices[i:i + batch_size] = min_indices
        nearest_neighbor_distances[i:i + batch_size] = min_distances

    return nearest_neighbor_indices, nearest_neighbor_distances

def cyclic_nn_map(features: torch.Tensor, masks: Dict[int, torch.Tensor],
                  latent_resolutions: List[int], device: torch.device) -> (Dict[int, torch.Tensor], Dict[int, torch.Tensor]):
    """
    Computes cyclic nearest–neighbor maps among all images in a batch.
    
    Args:
        features: Tensor of shape (batch, C, H, W).
        masks: Dictionary mapping latent resolution (e.g. 64) to binary masks.
        latent_resolutions: List of latent resolutions (e.g. [32, 64]).
        device: Torch device.
    
    Returns:
        Two dictionaries:
          - nn_map_dict: mapping each latent resolution to a tensor of shape (batch, batch, tgt_size**2)
          - nn_distances_dict: similarly for distances.
    """
    bsz = features.shape[0]
    nn_map_dict = {}
    nn_distances_dict = {}

    for tgt_size in latent_resolutions:
        nn_map = torch.empty(bsz, bsz, tgt_size ** 2, dtype=torch.long, device=device)
        nn_distances = torch.full((bsz, bsz, tgt_size ** 2), float('inf'), dtype=features.dtype, device=device)

        for i in range(bsz):
            for j in range(bsz):
                if i != j:
                    nn_idx, nn_dist = gen_nn_map(features[j], masks[tgt_size][j],
                                                 features[i], masks[tgt_size][i],
                                                 device, batch_size=None, tgt_size=tgt_size)
                    nn_map[i, j] = nn_idx
                    nn_distances[i, j] = nn_dist

        nn_map_dict[tgt_size] = nn_map
        nn_distances_dict[tgt_size] = nn_distances

    return nn_map_dict, nn_distances_dict

def anchor_nn_map(features: torch.Tensor, anchor_features: torch.Tensor,
                  masks: Dict[int, torch.Tensor], anchor_masks: Dict[int, torch.Tensor],
                  latent_resolutions: List[int], device: torch.device) -> (Dict[int, torch.Tensor], Dict[int, torch.Tensor]):
    """
    Computes nearest–neighbor mappings between main features and anchor features.
    
    Args:
        features: Main features tensor of shape (batch, C, H, W).
        anchor_features: Anchor features tensor of shape (anchor_batch, C, H, W).
        masks: Dictionary of masks for the main features.
        anchor_masks: Dictionary of masks for the anchor features.
        latent_resolutions: List of latent resolutions to compute (e.g. [32, 64]).
        device: Torch device.
    
    Returns:
        A tuple (nn_map_dict, nn_distances_dict) where each is a dictionary keyed by latent resolution.
    """
    bsz = features.shape[0]
    anchor_bsz = anchor_features.shape[0]
    nn_map_dict = {}
    nn_distances_dict = {}

    for tgt_size in latent_resolutions:
        nn_map = torch.empty(bsz, anchor_bsz, tgt_size ** 2, dtype=torch.long, device=device)
        nn_distances = torch.full((bsz, anchor_bsz, tgt_size ** 2), float('inf'), dtype=features.dtype, device=device)

        for i in range(bsz):
            for j in range(anchor_bsz):
                nn_idx, nn_dist = gen_nn_map(anchor_features[j], anchor_masks[tgt_size][j],
                                             features[i], masks[tgt_size][i],
                                             device, batch_size=None, tgt_size=tgt_size)
                nn_map[i, j] = nn_idx
                nn_distances[i, j] = nn_dist

        nn_map_dict[tgt_size] = nn_map
        nn_distances_dict[tgt_size] = nn_distances

    return nn_map_dict, nn_distances_dict
