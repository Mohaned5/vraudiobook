# Copyright (C) 2024 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.

import numpy as np
import torch
from collections import defaultdict
from diffusers.utils.import_utils import is_xformers_available
from typing import Optional, List

from .general_utils import get_dynamic_threshold

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class FeatureInjector:
    def __init__(
        self,
        nn_map,
        nn_distances,
        attn_masks,
        inject_range_alpha: List[Tuple[int, int, float]] = [(100, 300, 0.8)],
        swap_strategy: str = 'min',
        dist_thr: str = 'dynamic',
        inject_unet_parts: List[str] = ['up'],
    ):
        """
        Initializes the feature injector used for extended attention injection.

        Args:
            nn_map: Nearest neighbor map for the feature grid.
            nn_distances: Precomputed distances corresponding to nn_map.
            attn_masks: Binary attention masks.
            inject_range_alpha: A list of tuples (min_step, max_step, alpha) defining at which timesteps
                                and with what weight (alpha) to perform injection. For SD2.0 with ~1000 steps,
                                a default range of (100,300,0.8) is used.
            swap_strategy: Strategy for selecting features ('min', 'mean', or 'first').
            dist_thr: Either a numeric threshold or 'dynamic' to compute one from the data.
            inject_unet_parts: List of UNet parts where injection is allowed (e.g. ['up']).
        """
        self.nn_map = nn_map
        self.nn_distances = nn_distances
        self.attn_masks = attn_masks
        self.inject_range_alpha = inject_range_alpha if isinstance(inject_range_alpha, list) else [inject_range_alpha]
        self.swap_strategy = swap_strategy
        self.dist_thr = dist_thr
        self.inject_unet_parts = inject_unet_parts
        # Default latent resolution at which to apply injection (e.g. 64×64 for SD2.0 latents)
        self.inject_res = [64]

    def inject_outputs(self, output, curr_iter, output_res, extended_mapping, place_in_unet, anchors_cache=None):
        """
        Injects modified outputs into the UNet feature maps.

        Args:
            output: The current UNet output tensor.
            curr_iter: The current diffusion timestep (iteration).
            output_res: The spatial resolution of the output.
            extended_mapping: A binary mapping indicating which images to use for injection.
            place_in_unet: String indicating which part of the UNet (e.g. "up_3").
            anchors_cache: Optional cache to store intermediate outputs.
        Returns:
            The modified output tensor.
        """
        curr_unet_part = place_in_unet.split('_')[0]
        print(f"[DEBUG] Injecting outputs in {curr_unet_part} at iteration {curr_iter}.")

        # Only inject if the UNet part is allowed and the output resolution is one we target
        if (curr_unet_part not in self.inject_unet_parts) or (output_res not in self.inject_res):
            return output

        bsz = output.shape[0]
        nn_map = self.nn_map[output_res]
        nn_distances = self.nn_distances[output_res]
        attn_masks = self.attn_masks[output_res]
        vector_dim = output_res ** 2

        # Determine injection weight (alpha) based on current iteration
        alpha = next((alpha for min_range, max_range, alpha in self.inject_range_alpha 
                      if min_range <= curr_iter <= max_range), None)
        print(f"[DEBUG] Computed injection alpha: {alpha}")
        if alpha:
            old_output = output  # Consider cloning if you need to preserve the original values
            for i in range(bsz):
                if self.swap_strategy == 'min':
                    curr_mapping = extended_mapping[i]
                    # If the current image is not mapped to any other image, skip injection for this sample
                    if not torch.any(torch.cat([curr_mapping[:i], curr_mapping[i+1:]])):
                        continue
                    min_dists = nn_distances[i][curr_mapping].argmin(dim=0)
                    curr_nn_map = nn_map[i][curr_mapping][min_dists, torch.arange(vector_dim)]
                    curr_nn_distances = nn_distances[i][curr_mapping][min_dists, torch.arange(vector_dim)]
                    # Determine threshold (either dynamic or fixed)
                    threshold = get_dynamic_threshold(curr_nn_distances) if self.dist_thr == 'dynamic' else self.dist_thr
                    dist_mask = curr_nn_distances < threshold
                    final_mask_tgt = attn_masks[i] & dist_mask
                    injected_vals = old_output[curr_mapping][min_dists, curr_nn_map][final_mask_tgt]
                    output[i][final_mask_tgt] = alpha * injected_vals + (1 - alpha) * old_output[i][final_mask_tgt]

            if anchors_cache and anchors_cache.is_cache_mode():
                if place_in_unet not in anchors_cache.h_out_cache:
                    anchors_cache.h_out_cache[place_in_unet] = {}
                anchors_cache.h_out_cache[place_in_unet][curr_iter] = output

        return output

    def inject_anchors(self, output, curr_iter, output_res, extended_mapping, place_in_unet, anchors_cache):
        """
        Injects anchor features into the UNet output.

        Args:
            output: The current UNet output tensor.
            curr_iter: The current diffusion timestep (iteration).
            output_res: The spatial resolution of the output.
            extended_mapping: A binary mapping for injection.
            place_in_unet: A string indicating the UNet block (e.g. "up_3").
            anchors_cache: An AnchorCache instance storing cached anchor outputs.
        Returns:
            The modified output tensor.
        """
        curr_unet_part = place_in_unet.split('_')[0]
        if (curr_unet_part not in self.inject_unet_parts) or (output_res not in self.inject_res):
            return output

        bsz = output.shape[0]
        nn_map = self.nn_map[output_res]
        nn_distances = self.nn_distances[output_res]
        attn_masks = self.attn_masks[output_res]
        vector_dim = output_res ** 2

        alpha = next((alpha for min_range, max_range, alpha in self.inject_range_alpha 
                      if min_range <= curr_iter <= max_range), None)
        if alpha:
            anchor_outputs = anchors_cache.h_out_cache[place_in_unet][curr_iter]
            old_output = output  # Consider cloning if needed
            for i in range(bsz):
                if self.swap_strategy == 'min':
                    min_dists = nn_distances[i].argmin(dim=0)
                    curr_nn_map = nn_map[i][min_dists, torch.arange(vector_dim)]
                    curr_nn_distances = nn_distances[i][min_dists, torch.arange(vector_dim)]
                    threshold = get_dynamic_threshold(curr_nn_distances) if self.dist_thr == 'dynamic' else self.dist_thr
                    dist_mask = curr_nn_distances < threshold
                    final_mask_tgt = attn_masks[i] & dist_mask
                    injected_vals = anchor_outputs[min_dists, curr_nn_map][final_mask_tgt]
                    output[i][final_mask_tgt] = alpha * injected_vals + (1 - alpha) * old_output[i][final_mask_tgt]
        return output


class AnchorCache:
    def __init__(self):
        """
        Stores intermediate UNet features (“anchors”) for later injection.
        """
        self.input_h_cache = {}   # Mapping: {place_in_unet: {timestep: h_in}}
        self.h_out_cache = {}     # Mapping: {place_in_unet: {timestep: h_out}}
        self.anchors_last_mask = None
        self.dift_cache = None
        self.mode = 'cache'       # Modes: 'cache' or 'inject'

    def set_mode(self, mode):
        self.mode = mode

    def set_mode_inject(self):
        self.mode = 'inject'
        print("[DEBUG] AnchorCache mode switched to 'inject'.")

    def set_mode_cache(self):
        self.mode = 'cache'
        print("[DEBUG] AnchorCache mode switched to 'cache'.")

    def is_inject_mode(self):
        return self.mode == 'inject'

    def is_cache_mode(self):
        return self.mode == 'cache'

    def to_device(self, device):
        for key, value in self.input_h_cache.items():
            self.input_h_cache[key] = {k: v.to(device) for k, v in value.items()}
        for key, value in self.h_out_cache.items():
            self.h_out_cache[key] = {k: v.to(device) for k, v in value.items()}
        if self.anchors_last_mask:
            self.anchors_last_mask = {k: v.to(device) for k, v in self.anchors_last_mask.items()}
        if self.dift_cache is not None:
            self.dift_cache = self.dift_cache.to(device)


class QueryStore:
    def __init__(self, mode='store', t_range: List[int] = [0, 1000], strength_start=1, strength_end=1):
        """
        Initializes an empty QueryStore for caching and injecting query features.
        """
        self.query_store = defaultdict(list)
        self.mode = mode
        self.t_range = t_range
        self.strengthes = np.linspace(strength_start, strength_end, (t_range[1] - t_range[0]) + 1)

    def set_mode(self, mode):
        self.mode = mode

    def cache_query(self, query, place_in_unet: str):
        self.query_store[place_in_unet] = query

    def inject_query(self, query, place_in_unet, t):
        if self.t_range[0] <= t <= self.t_range[1]:
            relative_t = t - self.t_range[0]
            strength = self.strengthes[relative_t]
            new_query = strength * self.query_store[place_in_unet] + (1 - strength) * query
        else:
            new_query = query
        return new_query


class DIFTLatentStore:
    def __init__(self, steps: List[int], up_ft_indices: List[int]):
        """
        A simple latent feature store.
        Args:
            steps: List of diffusion timesteps at which to store features.
            up_ft_indices: List of indices (usually for upsample blocks) where features should be stored.
        """
        self.steps = steps
        self.up_ft_indices = up_ft_indices
        self.dift_features = {}

    def __call__(self, features: torch.Tensor, t: int, layer_index: int):
        if t in self.steps and layer_index in self.up_ft_indices:
            self.dift_features[f'{int(t)}_{layer_index}'] = features

    def copy(self):
        copy_dift = DIFTLatentStore(self.steps, self.up_ft_indices)
        for key, value in self.dift_features.items():
            copy_dift.dift_features[key] = value.clone()
        return copy_dift

    def reset(self):
        self.dift_features = {}
