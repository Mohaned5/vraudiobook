import torch
import torch.nn as nn
from einops import rearrange
from .modules import WarpAttn
from utils.pano import pad_pano, unpad_pano


class MultiViewBaseModel(nn.Module):
    def __init__(self, unet, pano_unet, pers_cn=None, pano_cn=None, pano_pad=True):
        super().__init__()

        self.unet = unet
        self.pano_unet = pano_unet
        self.pers_cn = pers_cn
        self.pano_cn = pano_cn
        self.pano_pad = pano_pad

        if self.unet is not None:
            self.cp_blocks_encoder = nn.ModuleList()
            for downsample_block in self.unet.down_blocks:
                if downsample_block.downsamplers is not None:
                    self.cp_blocks_encoder.append(
                        WarpAttn(downsample_block.downsamplers[-1].out_channels))

            self.cp_blocks_mid = WarpAttn(self.unet.mid_block.resnets[-1].out_channels)

            self.cp_blocks_decoder = nn.ModuleList()
            for upsample_block in self.unet.up_blocks:
                if upsample_block.upsamplers is not None:
                    self.cp_blocks_decoder.append(
                        WarpAttn(upsample_block.upsamplers[0].channels))

            self.trainable_parameters = [(
                list(self.cp_blocks_mid.parameters())
                + list(self.cp_blocks_decoder.parameters())
                + list(self.cp_blocks_encoder.parameters()),
                1.0
            )]

    def forward(
        self,
        latents,            # shape (b_lat, m_lat, C, H, W) or maybe (2*b_lat, m_lat, C,H,W) after CFG
        pano_latent,        # shape (b_pano, m_pano, C, H, W)
        timestep,           # shape (b_lat, m_lat), or (2*b_lat, m_lat), etc.
        prompt_embd,        # shape (b_lat, m_lat, L, C_text)
        pano_prompt_embd,   # shape (b_pano, m_pano, L, C_text)
        cameras,
        pers_layout_cond=None,
        pano_layout_cond=None
    ):
        # ------------------------
        # 1) Save original shapes
        # ------------------------
        if latents is not None:
            b_lat, m_lat, c, h, w = latents.shape
        else:
            b_lat, m_lat = 0, 0

        b_pano, m_pano, c2, h2, w2 = pano_latent.shape
        t_b, t_m = timestep.shape  # e.g. (2*b_lat, m_lat)

        # ------------
        # 2) Rearrange
        # ------------
        # latents => (b_lat*m_lat, c, h, w)
        if latents is not None:
            hidden_states = rearrange(latents, 'b m c h w -> (b m) c h w')
        else:
            hidden_states = None

        # cameras => each key => (b_lat*m_lat, ...)
        if cameras is not None:
            cameras = {k: rearrange(v, 'b m ... -> (b m) ...') for k, v in cameras.items()}

        # prompt_embd => (b_lat*m_lat, L, C_txt)
        if prompt_embd is not None:
            prompt_embd = rearrange(prompt_embd, 'b m l c -> (b m) l c')

        # pano_latent => (b_pano*m_pano, c, h, w)
        pano_hidden_states = rearrange(pano_latent, 'b m c h w -> (b m) c h w')

        # pano_prompt_embd => (b_pano*m_pano, L, C_txt)
        pano_prompt_embd = rearrange(pano_prompt_embd, 'b m l c -> (b m) l c')

        # timestep => flatten => shape (b_lat*m_lat,) or (2*b_lat*m_lat,)
        # But also we do:  pano_timestep = slice out the [:,0] if we treat that differently
        if self.unet is not None:
            pano_timestep = timestep[:, 0]          # shape (t_b, ) e.g. (2*b_lat)
            # flatten the rest => shape (t_b*m_t), typically (b_lat*m_lat) or (2*b_lat*m_lat)
            # if t_m=2 that means you might have 2 cameras or something
            # Just unify: (t_b*m_t) = (b_lat*m_lat) if no CFG duplication, or (2*b_lat*m_lat) if CFG.
            time_1d = rearrange(timestep, 'b m -> (b m)')
            t_emb = self.unet.time_proj(time_1d).to(self.unet.dtype)   # shape (b_lat*m_lat, 320)
            emb = self.unet.time_embedding(t_emb)                      # shape (b_lat*m_lat, 1280)
        else:
            pano_timestep = timestep

        # for the pano:
        pano_t_emb_1d = pano_timestep  # shape (t_b,) e.g. (2*b_lat) if CFG, or (b_lat).
        pano_t_emb = self.pano_unet.time_proj(pano_t_emb_1d).to(self.pano_unet.dtype)
        pano_emb = self.pano_unet.time_embedding(pano_t_emb)

        # ---------------------
        # 3) ControlNet Branch?
        # ---------------------
        if self.pers_cn is None:
            pers_layout_cond = None
        if self.pano_cn is None:
            pano_layout_cond = None

        # rearrange layout cond for pers
        if pers_layout_cond is not None and hidden_states is not None:
            pers_layout_cond = rearrange(pers_layout_cond, 'b m ... -> (b m) ...')
            (down_block_additional_residuals,
             mid_block_additional_residual) = self.pers_cn(
                 hidden_states,
                 time_1d,
                 encoder_hidden_states=prompt_embd,
                 controlnet_cond=pers_layout_cond,
                 return_dict=False,
             )
        else:
            down_block_additional_residuals = None
            mid_block_additional_residual = None

        # rearrange layout cond for pano
        if pano_layout_cond is not None:
            pano_layout_cond = rearrange(pano_layout_cond, 'b m ... -> (b m) ...')
            (pano_down_block_additional_residuals,
             pano_mid_block_additional_residual) = self.pano_cn(
                 pano_hidden_states,
                 pano_t_emb_1d,
                 encoder_hidden_states=pano_prompt_embd,
                 controlnet_cond=pano_layout_cond,
                 return_dict=False,
             )
        else:
            pano_down_block_additional_residuals = None
            pano_mid_block_additional_residual = None

        # ------------------
        # 4) UNet Forward
        # ------------------
        if self.unet is not None and hidden_states is not None:
            hidden_states = self.unet.conv_in(hidden_states)  # shape => (b_lat*m_lat, 320, H, W)

        if self.pano_pad:
            pano_hidden_states = pad_pano(pano_hidden_states, 1)
        pano_hidden_states = self.pano_unet.conv_in(pano_hidden_states)
        if self.pano_pad:
            pano_hidden_states = unpad_pano(pano_hidden_states, 1)

        # (a) downsample
        if hidden_states is not None:
            down_block_res_samples = (hidden_states,)
        else:
            down_block_res_samples = ()
        pano_down_block_res_samples = (pano_hidden_states,)

        # start looping over down blocks
        for i, downsample_block in enumerate(self.pano_unet.down_blocks):
            # has_cross_attention? => do cross-attn
            if getattr(downsample_block, 'has_cross_attention', False):
                for j in range(len(downsample_block.resnets)):
                    if hidden_states is not None:
                        hidden_states = self.unet.down_blocks[i].resnets[j](hidden_states, emb)
                        hidden_states = self.unet.down_blocks[i].attentions[j](
                            hidden_states, encoder_hidden_states=prompt_embd
                        ).sample
                        down_block_res_samples += (hidden_states,)

                    # Pano side
                    if self.pano_pad:
                        pano_hidden_states = pad_pano(pano_hidden_states, 2)
                    pano_hidden_states = self.pano_unet.down_blocks[i].resnets[j](
                        pano_hidden_states, pano_emb
                    )
                    if self.pano_pad:
                        pano_hidden_states = unpad_pano(pano_hidden_states, 2)
                    pano_hidden_states = self.pano_unet.down_blocks[i].attentions[j](
                        pano_hidden_states, encoder_hidden_states=pano_prompt_embd
                    ).sample
                    pano_down_block_res_samples += (pano_hidden_states,)

            # no cross-attn
            else:
                for j in range(len(downsample_block.resnets)):
                    if hidden_states is not None:
                        hidden_states = self.unet.down_blocks[i].resnets[j](hidden_states, emb)
                        down_block_res_samples += (hidden_states,)

                    if self.pano_pad:
                        pano_hidden_states = pad_pano(pano_hidden_states, 2)
                    pano_hidden_states = self.pano_unet.down_blocks[i].resnets[j](
                        pano_hidden_states, pano_emb
                    )
                    if self.pano_pad:
                        pano_hidden_states = unpad_pano(pano_hidden_states, 2)
                    pano_down_block_res_samples += (pano_hidden_states,)

            # any downsamplers in this block?
            if downsample_block.downsamplers is not None:
                for j in range(len(downsample_block.downsamplers)):
                    if hidden_states is not None:
                        hidden_states = self.unet.down_blocks[i].downsamplers[j](hidden_states)
                    if self.pano_pad:
                        pano_hidden_states = pad_pano(pano_hidden_states, 2)
                    pano_hidden_states = self.pano_unet.down_blocks[i].downsamplers[j](
                        pano_hidden_states
                    )
                    if self.pano_pad:
                        pano_hidden_states = unpad_pano(pano_hidden_states, 1)

                if hidden_states is not None:
                    down_block_res_samples += (hidden_states,)
                pano_down_block_res_samples += (pano_hidden_states,)

                # cross attention warp block
                if hidden_states is not None:
                    hidden_states, pano_hidden_states = self.cp_blocks_encoder[i](
                        hidden_states, pano_hidden_states, cameras
                    )

        # handle pers_layout_cond => apply additional controlnet residual
        if down_block_additional_residuals is not None:
            new_down_samples = ()
            for x_sample, x_resid in zip(down_block_res_samples, down_block_additional_residuals):
                new_down_samples += (x_sample + x_resid,)
            down_block_res_samples = new_down_samples

        # handle pano_layout_cond => apply additional controlnet residual
        if pano_down_block_additional_residuals is not None:
            new_pano_down = ()
            for x_sample, x_resid in zip(
                pano_down_block_res_samples, pano_down_block_additional_residuals
            ):
                new_pano_down += (x_sample + x_resid,)
            pano_down_block_res_samples = new_pano_down

        # (b) mid
        if hidden_states is not None:
            hidden_states = self.unet.mid_block.resnets[0](hidden_states, emb)
        if self.pano_pad:
            pano_hidden_states = pad_pano(pano_hidden_states, 2)
        pano_hidden_states = self.pano_unet.mid_block.resnets[0](pano_hidden_states, pano_emb)
        if self.pano_pad:
            pano_hidden_states = unpad_pano(pano_hidden_states, 2)

        for i in range(len(self.pano_unet.mid_block.attentions)):
            if hidden_states is not None:
                hidden_states = self.unet.mid_block.attentions[i](
                    hidden_states, encoder_hidden_states=prompt_embd
                ).sample
                hidden_states = self.unet.mid_block.resnets[i+1](hidden_states, emb)

            pano_hidden_states = self.pano_unet.mid_block.attentions[i](
                pano_hidden_states, encoder_hidden_states=pano_prompt_embd
            ).sample
            if self.pano_pad:
                pano_hidden_states = pad_pano(pano_hidden_states, 2)
            pano_hidden_states = self.pano_unet.mid_block.resnets[i+1](pano_hidden_states, pano_emb)
            if self.pano_pad:
                pano_hidden_states = unpad_pano(pano_hidden_states, 2)

        # any mid-level controlnet residual
        if self.unet is not None and mid_block_additional_residual is not None:
            hidden_states = hidden_states + mid_block_additional_residual
        if pano_mid_block_additional_residual is not None:
            pano_hidden_states = pano_hidden_states + pano_mid_block_additional_residual

        # cross-warp mid
        if self.unet is not None and hidden_states is not None:
            hidden_states, pano_hidden_states = self.cp_blocks_mid(
                hidden_states, pano_hidden_states, cameras
            )

        # (c) upsample
        for i, upsample_block in enumerate(self.pano_unet.up_blocks):
            if self.unet is not None and hidden_states is not None:
                # retrieve corresponding down-block residual samples for unet
                res_samples = down_block_res_samples[-len(upsample_block.resnets):]
                down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
            else:
                res_samples = ()

            pano_res_samples = pano_down_block_res_samples[-len(upsample_block.resnets):]
            pano_down_block_res_samples = pano_down_block_res_samples[:-len(upsample_block.resnets)]

            if getattr(upsample_block, 'has_cross_attention', False):
                # cross-attn
                for j in range(len(upsample_block.resnets)):
                    if self.unet is not None and hidden_states is not None:
                        res_hidden_states = res_samples[-1]
                        res_samples = res_samples[:-1]
                        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                        hidden_states = self.unet.up_blocks[i].resnets[j](hidden_states, emb)
                        hidden_states = self.unet.up_blocks[i].attentions[j](
                            hidden_states, encoder_hidden_states=prompt_embd
                        ).sample

                    pano_res_hidden_states = pano_res_samples[-1]
                    pano_res_samples = pano_res_samples[:-1]
                    pano_hidden_states = torch.cat(
                        [pano_hidden_states, pano_res_hidden_states], dim=1
                    )
                    if self.pano_pad:
                        pano_hidden_states = pad_pano(pano_hidden_states, 2)
                    pano_hidden_states = self.pano_unet.up_blocks[i].resnets[j](
                        pano_hidden_states, pano_emb
                    )
                    if self.pano_pad:
                        pano_hidden_states = unpad_pano(pano_hidden_states, 2)
                    pano_hidden_states = self.pano_unet.up_blocks[i].attentions[j](
                        pano_hidden_states, encoder_hidden_states=pano_prompt_embd
                    ).sample
            else:
                # no cross-attn
                for j in range(len(upsample_block.resnets)):
                    if self.unet is not None and hidden_states is not None:
                        res_hidden_states = res_samples[-1]
                        res_samples = res_samples[:-1]
                        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                        hidden_states = self.unet.up_blocks[i].resnets[j](hidden_states, emb)

                    pano_res_hidden_states = pano_res_samples[-1]
                    pano_res_samples = pano_res_samples[:-1]
                    pano_hidden_states = torch.cat(
                        [pano_hidden_states, pano_res_hidden_states], dim=1
                    )
                    if self.pano_pad:
                        pano_hidden_states = pad_pano(pano_hidden_states, 2)
                    pano_hidden_states = self.pano_unet.up_blocks[i].resnets[j](
                        pano_hidden_states, pano_emb
                    )
                    if self.pano_pad:
                        pano_hidden_states = unpad_pano(pano_hidden_states, 2)

            # upsamplers
            if upsample_block.upsamplers is not None:
                if self.unet is not None and hidden_states is not None:
                    hidden_states, pano_hidden_states = self.cp_blocks_decoder[i](
                        hidden_states, pano_hidden_states, cameras
                    )

                for j in range(len(upsample_block.upsamplers)):
                    if self.unet is not None and hidden_states is not None:
                        hidden_states = self.unet.up_blocks[i].upsamplers[j](hidden_states)
                    if self.pano_pad:
                        pano_hidden_states = pad_pano(pano_hidden_states, 1)
                    pano_hidden_states = self.pano_unet.up_blocks[i].upsamplers[j](
                        pano_hidden_states
                    )
                    if self.pano_pad:
                        pano_hidden_states = unpad_pano(pano_hidden_states, 2)

        # 4. post-process
        if self.unet is not None and hidden_states is not None:
            sample = self.unet.conv_norm_out(hidden_states)
            sample = self.unet.conv_act(sample)
            sample = self.unet.conv_out(sample)
            # reshape back => shape (b_lat, m_lat, C, H, W)
            sample = rearrange(sample, '(bm) c h w -> b m c h w', b=b_lat, m=m_lat)
        else:
            sample = None

        # pano side
        pano_sample = self.pano_unet.conv_norm_out(pano_hidden_states)
        pano_sample = self.pano_unet.conv_act(pano_sample)
        if self.pano_pad:
            pano_sample = pad_pano(pano_sample, 1)
        pano_sample = self.pano_unet.conv_out(pano_sample)
        if self.pano_pad:
            pano_sample = unpad_pano(pano_sample, 1)

        # shape => (b_pano, m_pano, C, H, W)
        pano_sample = rearrange(pano_sample, '(bm) c h w -> b m c h w', b=b_pano, m=m_pano)

        return sample, pano_sample
