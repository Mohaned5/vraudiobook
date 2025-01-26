import os
import json
import random
import numpy as np
import torch
from glob import glob
from PIL import Image
import cv2
import lightning as L

from einops import rearrange
from abc import abstractmethod
from utils.pano import Equirectangular, random_sample_camera, horizon_sample_camera, icosahedron_sample_camera
from external.Perspective_and_Equirectangular import mp2e
from .PanoDataset import PanoDataset, PanoDataModule, get_K_R
from torch.utils.data import DataLoader, Subset


class PanimeDataset(PanoDataset):

    def load_split(self, mode):
        if mode == 'predict':
            # 1) Read predict.json
            predict_file = os.path.join(self.data_dir, "predict.json")
            if not os.path.exists(predict_file):
                raise FileNotFoundError(f"Cannot find predict.json at {predict_file}")

            with open(predict_file, 'r') as f:
                all_data = json.load(f)

            # 2) Build new_data with 'scene_id' and 'view_id'
            new_data = []
            for sample in all_data:
                scene_id = sample["scene_id"]
                view_id = sample["view_id"]
                pano_prompt = sample.get("pano_prompt", "")
                # You can store any additional fields you want.
                # But at a minimum, matterport-style is scene_id + view_id.
                new_data.append({
                    "scene_id": scene_id,
                    "view_id": view_id,
                    "pano_prompt": pano_prompt
                })
            # new_data = new_data[:4] 
            return new_data

        else:
            split_file = os.path.join(self.data_dir, f"{mode}.json")
            if not os.path.exists(split_file):
                raise FileNotFoundError(f"Cannot find JSON split file: {split_file}")

            with open(split_file, 'r') as f:
                all_data = json.load(f)

            new_data = []
            for sample in all_data:
                # Create a unique pano_id from the file name (or any logic you prefer)
                pano_filename = os.path.basename(sample["pano"])
                pano_id = os.path.splitext(pano_filename)[0]

                # Define paths
                pano_path = os.path.join(self.data_dir, sample["pano"])
                images_paths = [os.path.join(self.data_dir, img) for img in sample["images"]]

                # Check if all paths exist
                if not os.path.exists(pano_path):
                    print(f"Skipping entry {pano_id}: pano file missing at {pano_path}")
                    continue

                if not all(os.path.exists(img_path) for img_path in images_paths):
                    print(f"Skipping entry {pano_id}: one or more images missing.")
                    continue

                # Add valid entries
                entry = {
                    "pano_id": pano_id,
                    "pano_path": pano_path,
                    "pano_prompt": sample.get("pano_prompt", ""),
                    "images_paths": images_paths,
                    "prompts": sample["prompts"],
                    "cameras_data": sample["cameras"]
                }
                new_data.append(entry)
                
            # new_data = new_data[:4] 
            return new_data

    def scan_results(self, result_dir):
        """
        If your results are saved under result_dir/<pano_id>/pano.png,
        then we can detect them by scanning all subfolders in result_dir.
        """
        folder_paths = glob(os.path.join(result_dir, '*'))
        # Extract folder names that might be valid pano_ids
        results_ids = {
            os.path.basename(p)
            for p in folder_paths
            if os.path.isdir(p)
        }
        return results_ids

    def get_data(self, idx):
        data = self.data[idx].copy()

        # 1) If in predict mode with repeated sampling
        if self.mode == 'predict':
            scene_id = data['scene_id']
            view_id = data['view_id']

            # Build a unique pano_id just like Matterport
            data['pano_id'] = f"{scene_id}_{view_id}"
            data['pano_prompt'] = data.get('pano_prompt', "")

        else:
            # 2) Possibly apply unconditioned training ratio
            if self.mode == 'train' and self.result_dir is None and random.random() < self.config['uncond_ratio']:
                data['pano_prompt'] = ""
                data['prompts'] = [""] * len(data['prompts'])

            # 3) Build 'cameras' dict in the format PanFusion expects
            cam_data = data['cameras_data']
            FoV = torch.as_tensor(np.array(cam_data['FoV'][0], dtype=np.float32))
            theta = torch.as_tensor(np.array(cam_data['theta'][0], dtype=np.float32))
            phi = torch.as_tensor(np.array(cam_data['phi'][0], dtype=np.float32))
            
            cameras = {
                # If you want each perspective to be 256×256, set these to pers_resolution
                # or keep them as you like. Here we hard-code to 512 as an example.
                'height': 256,
                'width': 256,
                'FoV': FoV,
                'theta': theta,
                'phi': phi,
            }

            # Compute intrinsics & extrinsics
            Ks, Rs = [], []
            for f, t, p in zip(FoV, theta, phi):
                K, R = get_K_R(
                    f.item(), t.item(), p.item(),
                    self.config['pers_resolution'],
                    self.config['pers_resolution']
                )
                Ks.append(K)
                Rs.append(R)
            cameras['K'] = torch.as_tensor(np.stack(Ks).astype(np.float32))
            cameras['R'] = torch.as_tensor(np.stack(Rs).astype(np.float32))

            # 4) Save everything the pipeline expects
            data['prompt'] = data['prompts']
            data['cameras'] = cameras
            data['height'] = self.config['pano_height']
            data['width'] = self.config['pano_height'] * 2  # typical equirect (2:1 ratio)

        # # 5) Provide the path for the pano
        # if self.mode != 'predict':
        #     data['pano_path'] = data['pano_path']

        # 6) If there's a results directory, set path for predicted pano
        if self.result_dir is not None:
            data['pano_pred_path'] = os.path.join(self.result_dir, data['pano_id'], 'pano.png')

        return data


class PanimeDataModule(PanoDataModule):
    """
    A stripped-down data module focusing only on training (and optionally predict).
    """

    def __init__(
        self,
        data_dir: str = 'data/Panime',
        **kwargs
    ):
        super().__init__(data_dir=data_dir, **kwargs)
        self.save_hyperparameters()
        # Use PanimeDataset instead of the base PanoDataset
        self.dataset_cls = PanimeDataset

    def setup(self, stage=None):
        if stage in ('fit', None):
            self.train_dataset = self.dataset_cls(self.hparams, mode='train')

            # -----------------------------
            # 1) Create a stabilized subset
            # -----------------------------
            total_len = len(self.train_dataset)
            subset_len = int(0.25 * total_len)  # 15% of train set for example

            # Fix a random seed for reproducibility
            g = torch.Generator()
            g.manual_seed(1234)

            # Shuffle all indices in a reproducible way
            indices = torch.randperm(total_len, generator=g).tolist()
            stabilized_indices = indices[:subset_len]

            # Create the subset and store it
            self.train_stabilized_subset = Subset(self.train_dataset, stabilized_indices)

            # Create a dedicated DataLoader for it
            self.train_stabilized_loader = DataLoader(
                self.train_stabilized_subset,
                batch_size=self.hparams.batch_size,
                shuffle=False,  # No shuffle => consistent ordering
                num_workers=self.hparams.num_workers,
                drop_last=False
            )

        if stage in ('fit', 'validate', None):
            self.val_dataset = self.dataset_cls(self.hparams, mode='val')

        if stage in ('test', None):
            self.test_dataset = self.dataset_cls(self.hparams, mode='test')

        # Predict dataset (optional)
        if stage in ('predict', None):
            self.predict_dataset = self.dataset_cls(self.hparams, mode='predict')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            drop_last=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            drop_last=False
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            drop_last=False
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=False
        )
