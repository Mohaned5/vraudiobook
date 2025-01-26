from .PanoGenerator import PanoGenerator
from ..modules.utils import tensor_to_image
from .MVGenModel import MultiViewBaseModel
import torch
import os
from PIL import Image
from external.Perspective_and_Equirectangular import e2p
from einops import rearrange
from lightning.pytorch.utilities import rank_zero_only
import lpips
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as T


class PanFusion(PanoGenerator):
    def __init__(
            self,
            use_pers_prompt: bool = True,
            use_pano_prompt: bool = True,
            copy_pano_prompt: bool = True,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

       # in your __init__:
        self.lpips_model = lpips.LPIPS(net='alex')
        # Mark them as eval and not trainable
        self.lpips_model.eval()
        for param in self.lpips_model.parameters():
            param.requires_grad = False

        self.fid_metric = FrechetInceptionDistance(feature=2048, normalize=True)
        for param in self.fid_metric.parameters():
            param.requires_grad = False
        
        self.fid_transform = T.Compose([
            T.Resize(299),           
            T.CenterCrop(299),
            T.ToTensor(),             
        ])

        self._val_lpips = []

        

    def instantiate_model(self):
        pano_unet, cn = self.load_pano()
        unet, pers_cn = self.load_pers()
        self.mv_base_model = MultiViewBaseModel(unet, pano_unet, pers_cn, cn, self.hparams.unet_pad)
        if not self.hparams.layout_cond:
            self.trainable_params.extend(self.mv_base_model.trainable_parameters)

    def init_noise(self, bs, equi_h, equi_w, pers_h, pers_w, cameras, device):
        cameras = {k: rearrange(v, 'b m ... -> (b m) ...') for k, v in cameras.items()}
        m = len(cameras['FoV']) // bs
        pano_noise = torch.randn(
            bs, 1, 4, equi_h, equi_w, device=device)
        pano_noises = pano_noise.expand(-1, m, -1, -1, -1)
        pano_noises = rearrange(pano_noises, 'b m c h w -> (b m) c h w')
        noise = e2p(
            pano_noises,
            cameras['FoV'], cameras['theta'], cameras['phi'],
            (pers_h, pers_w), mode='nearest')
        noise = rearrange(noise, '(b m) c h w -> b m c h w', b=bs, m=m)
        # noise_sample = noise[0, 0, :3]
        # pano_noise_sample = pano_noise[0, 0, :3]
        return pano_noise, noise

    def embed_prompt(self, batch, num_cameras):
        if self.hparams.use_pers_prompt:
            pers_prompt = self.get_pers_prompt(batch)
            pers_prompt_embd = self.encode_text(pers_prompt)
            pers_prompt_embd = rearrange(pers_prompt_embd, '(b m) l c -> b m l c', m=num_cameras)
        else:
            pers_prompt = ''
            pers_prompt_embd = self.encode_text(pers_prompt)
            pers_prompt_embd = pers_prompt_embd[:, None].repeat(1, num_cameras, 1, 1)

        if self.hparams.use_pano_prompt:
            pano_prompt = self.get_pano_prompt(batch)
        else:
            pano_prompt = ''
        pano_prompt_embd = self.encode_text(pano_prompt)
        pano_prompt_embd = pano_prompt_embd[:, None]

        return pers_prompt_embd, pano_prompt_embd

    def training_step(self, batch, batch_idx):
        device = batch['images'].device
        latents = self.encode_image(batch['images'], self.vae)
        b, m, c, h, w = latents.shape

        pano_pad = self.pad_pano(batch['pano'])
        pano_latent_pad = self.encode_image(pano_pad, self.vae)
        pano_latent = self.unpad_pano(pano_latent_pad, latent=True)
        # # test encoded pano latent
        # pano_pad = ((pano_pad[0, 0] + 1) * 127.5).cpu().numpy().astype(np.uint8)
        # pano = ((batch['pano'][0, 0] + 1) * 127.5).cpu().numpy().astype(np.uint8)
        # pano_decode = self.decode_latent(pano_latent, self.vae)[0, 0]

        t = torch.randint(0, self.scheduler.config.num_train_timesteps,
                          (b,), device=latents.device).long()
        pers_prompt_embd, pano_prompt_embd = self.embed_prompt(batch, m)
        pano_noise, noise = self.init_noise(
            b, *pano_latent.shape[-2:], h, w, batch['cameras'], device)

        noise_z = self.scheduler.add_noise(latents, noise, t)
        pano_noise_z = self.scheduler.add_noise(pano_latent, pano_noise, t)
        t = t[:, None].repeat(1, m)

        denoise, pano_denoise = self.mv_base_model(
            noise_z, pano_noise_z, t, pers_prompt_embd, pano_prompt_embd, batch['cameras'],
            batch.get('images_layout_cond'), batch.get('pano_layout_cond'))

        # eps mode
        loss_pers = torch.nn.functional.mse_loss(denoise, noise)
        loss_pano = torch.nn.functional.mse_loss(pano_denoise, pano_noise)
        loss = loss_pers + loss_pano
        self.log('train/loss', loss, prog_bar=False)
        self.log('train/loss_pers', loss_pers, prog_bar=True)
        self.log('train/loss_pano', loss_pano, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        # Put the model in eval mode if needed
        self.eval()
        
        # Optionally fix random seed if you want consistent noise
        torch.manual_seed(9999)

        total_loss = 0.0
        count = 0

        # Get your 'stabilized' DataLoader from the DataModule
        stabilized_loader = self.trainer.datamodule.train_stabilized_loader

        # No gradients
        with torch.no_grad():
            for batch_idx, batch in enumerate(stabilized_loader):
                # 1) Manually move the batch to the correct device
                batch = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)
                
                # 2) Now call training_step (or your custom function)
                loss = self.training_step(batch, batch_idx)
                total_loss += loss.item()
                count += 1

        avg_loss = total_loss / max(count, 1)
        self.log("train_stabilized", avg_loss, sync_dist=True)
        self.train()  # switch back to training mode



    @torch.no_grad()
    def forward_cls_free(self, latents, pano_latent, timestep, prompt_embd, pano_prompt_embd, batch, pano_layout_cond=None):
        latents, pano_latent, timestep, cameras, images_layout_cond, pano_layout_cond = self.gen_cls_free_guide_pair(
            latents, pano_latent, timestep, batch['cameras'],
            batch.get('images_layout_cond'), pano_layout_cond)

        noise_pred, pano_noise_pred = self.mv_base_model(
            latents, pano_latent, timestep, prompt_embd, pano_prompt_embd, cameras,
            images_layout_cond, pano_layout_cond)

        noise_pred, pano_noise_pred = self.combine_cls_free_guide_pred(noise_pred, pano_noise_pred)

        return noise_pred, pano_noise_pred

    def rotate_latent(self, pano_latent, cameras, degree=None):
        if degree is None:
            degree = self.hparams.rot_diff
        if degree % 360 == 0:
            return pano_latent, cameras

        pano_latent = super().rotate_latent(pano_latent, degree)
        cameras = cameras.copy()
        cameras['theta'] = (cameras['theta'] + degree) % 360
        return pano_latent, cameras

    @torch.no_grad()
    def inference(self, batch):
        bs, m = batch['cameras']['height'].shape[:2]
        h, w = batch['cameras']['height'][0, 0].item(), batch['cameras']['width'][0, 0].item()

        equi_h = int(batch['height'][0] // 8)
        equi_w = int(batch['width'][0] // 8)
        device = self.device

        pano_latent, latents = self.init_noise(
            bs, equi_h, equi_w, h//8, h//8, batch['cameras'], device)

        pers_prompt_embd, pano_prompt_embd = self.embed_prompt(batch, m)
        prompt_null = self.encode_text('')[:, None]
        pano_prompt_embd = torch.cat([prompt_null, pano_prompt_embd])
        prompt_null = prompt_null.repeat(1, m, 1, 1)
        pers_prompt_embd = torch.cat([prompt_null, pers_prompt_embd])

        self.scheduler.set_timesteps(self.hparams.diff_timestep, device=device)
        timesteps = self.scheduler.timesteps

        pano_layout_cond = batch.get('pano_layout_cond')

        curr_rot = 0
        for i, t in enumerate(timesteps):
            timestep = torch.cat([t[None, None]]*m, dim=1)

            pano_latent, batch['cameras'] = self.rotate_latent(pano_latent, batch['cameras'])
            curr_rot += self.hparams.rot_diff

            if self.hparams.layout_cond:
                pano_layout_cond = super().rotate_latent(pano_layout_cond)
            else:
                pano_layout_cond = None
            noise_pred, pano_noise_pred = self.forward_cls_free(
                latents, pano_latent, timestep, pers_prompt_embd, pano_prompt_embd, batch, pano_layout_cond)

            latents = self.scheduler.step(
                noise_pred, t, latents).prev_sample
            pano_latent = self.scheduler.step(
                pano_noise_pred, t, pano_latent).prev_sample

        pano_latent, batch['cameras'] = self.rotate_latent(pano_latent, batch['cameras'], -curr_rot)

        images_pred = self.decode_latent(latents, self.vae)
        images_pred = tensor_to_image(images_pred)

        pano_latent_pad = self.pad_pano(pano_latent, latent=True)
        pano_pred_pad = self.decode_latent(pano_latent_pad, self.vae)
        pano_pred = self.unpad_pano(pano_pred_pad)
        pano_pred = tensor_to_image(pano_pred)

        # # test encoded pano latent
        # img1 = self.decode_latent(pano_latent, self.vae).squeeze()
        # img1 = np.roll(img1, img1.shape[0]//2, axis=0)
        # img1 = np.roll(img1, img1.shape[1]//2, axis=1)
        # img2 = pano_pred.squeeze()
        # img2 = np.roll(img2, img2.shape[0]//2, axis=0)
        # img2 = np.roll(img2, img2.shape[1]//2, axis=1)

        return images_pred, pano_pred

    def to01(self, x: torch.Tensor) -> torch.Tensor:
        """
        Converts a tensor in [-1, 1] to [0, 1].
        """
        return (x + 1.) * 0.5

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        device = batch['images'].device
        total_loss = 0.0
        total_loss_pers = 0.0
        total_loss_pano = 0.0
        lpips_scores = []
        
        # Iterate over each sample in the batch
        for i in range(batch['images'].size(0)):
            # Extract the i-th sample and add a new batch dimension
            single_batch = {k: v[i].unsqueeze(0) for k, v in batch.items()}
            
            # 1) Encode the ground-truth perspective images
            latents = self.encode_image(single_batch['images'], self.vae)  # shape: (1, m, c, h, w)
            # 2) Encode the ground-truth pano
            pano_pad = self.pad_pano(single_batch['pano'])
            pano_latent_pad = self.encode_image(pano_pad, self.vae)
            pano_latent = self.unpad_pano(pano_latent_pad, latent=True)
            
            # 3) Sample a random t for the sample
            b, m, c, h, w = latents.shape
            t = torch.randint(0, self.scheduler.config.num_train_timesteps, (b,), device=device).long()
            
            # 4) Create prompts
            pers_prompt_embd, pano_prompt_embd = self.embed_prompt(single_batch, m)
            
            # 5) Initialize noise
            pano_noise, noise = self.init_noise(
                b, pano_latent.shape[-2], pano_latent.shape[-1], h, w, single_batch['cameras'], device
            )
            
            # 6) Add noise
            noise_z = self.scheduler.add_noise(latents, noise, t)
            pano_noise_z = self.scheduler.add_noise(pano_latent, pano_noise, t)
            t = t[:, None].repeat(1, m)  # shape: (1, m)
            
            # 7) Forward pass
            denoise, pano_denoise = self.mv_base_model(
                noise_z, pano_noise_z, t, pers_prompt_embd, pano_prompt_embd,
                single_batch['cameras'], single_batch.get('images_layout_cond'), single_batch.get('pano_layout_cond')
            )
            
            # 8) Compute MSE losses
            loss_pers = torch.nn.functional.mse_loss(denoise, noise)
            loss_pano = torch.nn.functional.mse_loss(pano_denoise, pano_noise)
            val_loss = loss_pers + loss_pano
            
            # Accumulate losses
            total_loss += val_loss.item()
            total_loss_pers += loss_pers.item()
            total_loss_pano += loss_pano.item()
            
            # --- Compute and Accumulate LPIPS and FID Metrics ---
            # Convert predictions and ground truth to images
            images_pred, pano_pred = self.inference(single_batch)
            
            for j in range(m):
                # Process perspective images
                pred_img = images_pred[0, j].cpu().numpy()
                gt_img = single_batch['images'][0, j].cpu()
                
                # Convert pred_img (NumPy) to Torch and permute if necessary
                pred_img_torch = torch.from_numpy(pred_img).float()
                if pred_img_torch.ndim == 3 and pred_img_torch.shape[-1] == 3:
                    pred_img_torch = pred_img_torch.permute(2, 0, 1)
                
                pred_img_torch = pred_img_torch.to(device)
                gt_img_torch = gt_img.to(device).float()
                
                # Compute LPIPS
                lpips_val = self.lpips_model(
                    pred_img_torch.unsqueeze(0),  # (1, 3, H, W)
                    gt_img_torch.unsqueeze(0)     # (1, 3, H, W)
                )
                lpips_scores.append(lpips_val.item())
                
                # Compute FID
                pred_img_01 = self.to01(pred_img_torch).clamp(0, 1).cpu()
                gt_img_01 = self.to01(gt_img_torch).clamp(0, 1).cpu()
                
                pred_pil = transforms.ToPILImage()(pred_img_01)
                gt_pil = transforms.ToPILImage()(gt_img_01)
                
                pred_img_resized = self.fid_transform(pred_pil).to(device)
                gt_img_resized = self.fid_transform(gt_pil).to(device)
                
                self.fid_metric.update(pred_img_resized.unsqueeze(0), real=False)
                self.fid_metric.update(gt_img_resized.unsqueeze(0), real=True)
            
            # Process pano image
            pano_pred_img = pano_pred[0, 0].cpu().numpy()
            pano_gt_img = single_batch['pano'][0, 0].cpu()
            
            pano_pred_torch = torch.from_numpy(pano_pred_img).float()
            if pano_pred_torch.ndim == 3 and pano_pred_torch.shape[-1] == 3:
                pano_pred_torch = pano_pred_torch.permute(2, 0, 1)
            
            pano_pred_torch = pano_pred_torch.to(device)
            pano_gt_torch = pano_gt_img.to(device).float()
            
            # Compute LPIPS for pano
            pano_lpips = self.lpips_model(
                pano_pred_torch.unsqueeze(0),
                pano_gt_torch.unsqueeze(0)
            )
            lpips_scores.append(pano_lpips.item())
            
            # Compute FID for pano
            pano_pred_01 = self.to01(pano_pred_torch).clamp(0, 1).cpu()
            pano_gt_01 = self.to01(pano_gt_torch).clamp(0, 1).cpu()
            
            pano_pred_pil = transforms.ToPILImage()(pano_pred_01)
            pano_gt_pil = transforms.ToPILImage()(pano_gt_01)
            
            pano_pred_resized = self.fid_transform(pano_pred_pil).to(device)
            pano_gt_resized = self.fid_transform(pano_gt_pil).to(device)
            
            self.fid_metric.update(pano_pred_resized.unsqueeze(0), real=False)
            self.fid_metric.update(pano_gt_resized.unsqueeze(0), real=True)
        
        # Compute average losses
        avg_loss = total_loss / batch['images'].size(0)
        avg_loss_pers = total_loss_pers / batch['images'].size(0)
        avg_loss_pano = total_loss_pano / batch['images'].size(0)
        
        # Compute average LPIPS
        if lpips_scores:
            avg_lpips = sum(lpips_scores) / len(lpips_scores)
        else:
            avg_lpips = float('nan')
        
        # Compute FID for the epoch (optional: move to on_validation_epoch_end)
        try:
            fid_score = self.fid_metric.compute().item()
        except Exception as e:
            self.print(f"Error computing FID: {e}")
            fid_score = float('nan')
        
        # Log the metrics
        self.log('val/loss', avg_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/loss_pers', avg_loss_pers, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/loss_pano', avg_loss_pano, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/lpips', avg_lpips, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/fid', fid_score, prog_bar=True, on_step=False, on_epoch=True)
        
        # Reset FID metric for the next batch
        self.fid_metric.reset()
        
        return avg_loss


    
    def on_validation_epoch_end(self):
        # --- Aggregate LPIPS Scores ---
        if self._val_lpips:
            epoch_avg_lpips = sum(self._val_lpips) / len(self._val_lpips)
        else:
            epoch_avg_lpips = float('nan')
        self.log('val/lpips_epoch', epoch_avg_lpips, prog_bar=True)

        # --- Compute FID for the Epoch ---
        try:
            epoch_fid = self.fid_metric.compute().item()
        except Exception as e:
            self.print(f"Error computing FID: {e}")
            epoch_fid = float('nan')
        self.log('val/fid_epoch', epoch_fid, prog_bar=True)

        # --- Reset Metrics for Next Epoch ---
        self._val_lpips.clear()
        self.fid_metric.reset()

        if self.trainer and self.trainer.is_global_zero:
            checkpoint_dir = "checkpoints_val_epoch"
            os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"epoch_{self.current_epoch}.ckpt"
            )
            self.trainer.save_checkpoint(checkpoint_path)
            print(f"Saved checkpoint at end of validation epoch: {checkpoint_path}")



    def inference_and_save(self, batch, output_dir, ext='png'):
        prompt_path = os.path.join(output_dir, 'prompt.txt')
        print(f"Debug: Using prompt => {batch['pano_prompt']}")
        if os.path.exists(prompt_path):
            return

        _, pano_pred = self.inference(batch)

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"pano.{ext}")
        im = Image.fromarray(pano_pred[0, 0])
        im.save(path)

        with open(prompt_path, 'w') as f:
            f.write(batch['pano_prompt'][0]+'\n')

    @torch.no_grad()
    @rank_zero_only
    def log_val_image(self, images_pred, images, pano_pred, pano, pano_prompt,
                      images_layout_cond=None, pano_layout_cond=None):
        log_dict = {f"val/{k}_pred": v for k, v in self.temp_wandb_images(
            images_pred, pano_pred, None, pano_prompt).items()}
        log_dict.update({f"val/{k}_gt": v for k, v in self.temp_wandb_images(
            images, pano, None, pano_prompt).items()})
        if images_layout_cond is not None and pano_layout_cond is not None:
            log_dict.update({f"val/{k}_layout_cond": v for k, v in self.temp_wandb_images(
                images_layout_cond, pano_layout_cond, None, pano_prompt).items()})
        self.logger.experiment.log(log_dict)

    def temp_wandb_images(self, images, pano, prompt=None, pano_prompt=None):
        log_dict = {}
        pers = []
        for m_i in range(images.shape[1]):
            pers.append(self.temp_wandb_image(
                images[0, m_i], prompt[m_i][0] if prompt else None))
        log_dict['pers'] = pers

        log_dict['pano'] = self.temp_wandb_image(
            pano[0, 0], pano_prompt[0] if pano_prompt else None)
        return log_dict
