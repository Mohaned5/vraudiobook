# Copyright (C) 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License
# 
# Copyright (c) 2023 AttendAndExcite
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Union, List
import torch
from collections import defaultdict
import numpy as np
from PIL import Image
from IPython.display import display
import torch.nn.functional as F

from .general_utils import attn_map_to_binary


def view_images(
    images: Union[np.ndarray, List],
    num_rows: int = 1,
    offset_ratio: float = 0.02,
    display_image: bool = True,
    downscale_rate: int = None
) -> Image.Image:
    """Displays a list of images in a grid."""
    if isinstance(images, list):
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    canvas = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            canvas[
                i * (h + offset): i * (h + offset) + h,
                j * (w + offset): j * (w + offset) + w
            ] = images[i * num_cols + j]

    pil_img = Image.fromarray(canvas)
    if downscale_rate:
        new_size = (int(pil_img.size[0] // downscale_rate), int(pil_img.size[1] // downscale_rate))
        pil_img = pil_img.resize(new_size)
    if display_image:
        display(pil_img)
    return pil_img


class AttentionStore:
    def __init__(self, attention_store_kwargs: dict):
        """
        Initialize an empty AttentionStore.
        
        Args:
            attention_store_kwargs: Dictionary containing parameters for attention storage,
              e.g. 'attn_res', 'token_indices', 'mask_dropout', etc.
        """
        self.attn_res = attention_store_kwargs.get('attn_res', (32, 32))
        self.token_indices = attention_store_kwargs['token_indices']
        bsz = self.token_indices.size(1)
        self.mask_background_query = attention_store_kwargs.get('mask_background_query', False)
        self.original_attn_masks = attention_store_kwargs.get('original_attn_masks', None)
        self.extended_mapping = attention_store_kwargs.get('extended_mapping', torch.ones(bsz, bsz).bool())
        self.mask_dropout = attention_store_kwargs.get('mask_dropout', 0.0)
        torch.manual_seed(0)  # For dropout mask reproducibility

        self.curr_iter = 0
        self.ALL_RES = [32, 64]
        self.step_store = defaultdict(list)
        self.attn_masks = {res: None for res in self.ALL_RES}
        self.last_mask = {res: None for res in self.ALL_RES}
        self.last_mask_dropout = {res: None for res in self.ALL_RES}

    def __call__(self, attn: torch.Tensor, is_cross: bool, place_in_unet: str, attn_heads: int):
        # If cross attention and the attention tensor has the expected size,
        # process and store the attention map.
        if is_cross and attn.shape[1] == np.prod(self.attn_res):
            guidance_attention = attn[attn.size(0) // 2:]
            batched_attention = guidance_attention.reshape(
                [guidance_attention.shape[0] // attn_heads, attn_heads, *guidance_attention.shape[1:]]
            ).mean(dim=1)
            self.step_store[place_in_unet].append(batched_attention)

    def reset(self):
        self.step_store = defaultdict(list)
        self.attn_masks = {res: None for res in self.ALL_RES}
        self.last_mask = {res: None for res in self.ALL_RES}
        self.last_mask_dropout = {res: None for res in self.ALL_RES}
        torch.cuda.empty_cache()

    def aggregate_last_steps_attention(self) -> torch.Tensor:
        """Aggregate the attention maps from the last 20 steps for each layer."""
        attention_maps = torch.cat(
            [torch.stack(x[-20:]) for x in self.step_store.values()]
        ).mean(dim=0)
        bsz, wh, _ = attention_maps.shape

        agg_attn_maps = []
        for i in range(bsz):
            curr_maps = []
            for token_idx in self.token_indices:
                if token_idx[i] != -1:
                    curr_maps.append(attention_maps[i, :, token_idx[i]].view(*self.attn_res))
            agg_attn_maps.append(torch.stack(curr_maps))
        # Upsample each batch item's attention maps to each target resolution
        for tgt_size in self.ALL_RES:
            pixels = tgt_size ** 2
            upsampled_maps = [torch.nn.functional.interpolate(x.unsqueeze(1), size=tgt_size, mode='bilinear').squeeze(1)
                              for x in agg_attn_maps]
            attn_masks = []
            for batch_item_map in upsampled_maps:
                masks = [torch.from_numpy(attn_map_to_binary(concept_map, 1.)).to(attention_maps.device).bool().view(-1)
                         for concept_map in batch_item_map]
                masks = torch.stack(masks, dim=0).max(dim=0).values
                attn_masks.append(masks)
            attn_masks = torch.stack(attn_masks)
            self.last_mask[tgt_size] = attn_masks.clone()
            # Apply dropout to the mask if within the first 1000 iterations.
            if self.curr_iter < 1000:
                rand_mask = (torch.rand_like(attn_masks.float()) < self.mask_dropout)
                attn_masks[rand_mask] = False
            self.last_mask_dropout[tgt_size] = attn_masks.clone()

        return attention_maps

    def get_attn_mask_bias(self, tgt_size: int, bsz: int = None) -> torch.Tensor:
        attn_mask = self.attn_masks[tgt_size] if self.original_attn_masks is None else self.original_attn_masks[tgt_size]
        if attn_mask is None:
            return None
        attn_bias = torch.zeros_like(attn_mask, dtype=torch.float16)
        attn_bias[~attn_mask] = float('-inf')
        if bsz and bsz != attn_bias.shape[0]:
            attn_bias = attn_bias.repeat(bsz // attn_bias.shape[0], 1, 1)
        return attn_bias

    def get_extended_attn_mask_instance(self, width: int, i: int) -> torch.Tensor:
        attn_mask = self.last_mask_dropout[width]
        if attn_mask is None:
            return None
        n_patches = width ** 2
        output_attn_mask = torch.zeros((attn_mask.shape[0] * attn_mask.shape[1],), device=attn_mask.device, dtype=torch.bool)
        for j in range(attn_mask.shape[0]):
            if i == j:
                output_attn_mask[j * n_patches:(j + 1) * n_patches] = 1
            else:
                if self.extended_mapping[i, j]:
                    if not self.mask_background_query:
                        output_attn_mask[j * n_patches:(j + 1) * n_patches] = attn_mask[j].unsqueeze(0)
                    else:
                        raise NotImplementedError('mask_background_query is not supported anymore')
        return output_attn_mask


__all__ = ["view_images", "AttentionStore"]
