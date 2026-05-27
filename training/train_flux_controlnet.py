"""
FLUX.1-Fill + ControlNet training for hand inpainting.

Architecture:
  - FLUX.1-Fill transformer: frozen  (handles masked-image inpainting natively)
  - FluxControlNetModel: trainable   (injects skeleton-on-UV condition)

Conditioning:
  - Inpainting: masked_image + mask concatenated in latent space (native to Fill)
  - ControlNet: skeleton overlaid on UV map (composite RGB, 256x256)

Usage:
    accelerate launch training/train_flux_controlnet.py --config configs/train_flux.yaml
"""

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from diffusers import (
    AutoencoderKL,
    FluxControlNetModel,
    FluxTransformer2DModel,
)
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from transformers import BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.hand_dataset import make_dataloaders

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_frozen_components(model_id: str, dtype: torch.dtype, device: str):
    """Load VAE and transformer from FLUX.1-Fill; freeze everything.

    The transformer is loaded in NF4 4-bit (~6 GB vs ~24 GB in bf16) since
    it is fully frozen — no gradients, no optimizer states needed.
    """
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=dtype
    ).to(device)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=bnb_config,
        torch_dtype=dtype,
    )
    # Quantized models are placed on device automatically by bitsandbytes
    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    return vae, transformer


def encode_images(vae, images: torch.Tensor) -> torch.Tensor:
    """Encode (B,3,H,W) in [-1,1] → packed latents (B, H/16*W/16, 64)."""
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample()
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    # Pack 2x2 spatial patches: (B,16,H/8,W/8) → (B,H/16*W/16, 64)
    return pack_latents(latents)


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) → (B, H/2 * W/2, C*4)  FLUX 2x2 patch packing."""
    B, C, H, W = latents.shape
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5).reshape(B, (H // 2) * (W // 2), C * 4)
    return latents


def pack_mask(mask: torch.Tensor, vae_scale_factor: int = 8) -> torch.Tensor:
    """
    Encode the binary mask (B,1,H,W) at IMAGE resolution the same way
    FluxFillPipeline does: pixel-fold into (B, vae_scale_factor², h_lat, w_lat)
    then 2×2-pack → (B, h_lat/2 * w_lat/2, vae_scale_factor² × 4).

    For 256×256 input with scale_factor=8:
      (B,1,256,256) → (B,64,32,32) → (B,256,256)
    """
    B, _, H, W = mask.shape
    # Remove channel dim, keeping (B, H, W)
    mask = mask[:, 0]
    h_lat = H // vae_scale_factor   # latent height (32 for 256px)
    w_lat = W // vae_scale_factor   # latent width  (32 for 256px)
    # Fold 8×8 pixel blocks into channels: (B, h_lat, 8, w_lat, 8)
    mask = mask.view(B, h_lat, vae_scale_factor, w_lat, vae_scale_factor)
    # (B, 8, 8, h_lat, w_lat) → (B, 64, h_lat, w_lat)
    mask = mask.permute(0, 2, 4, 1, 3).reshape(B, vae_scale_factor ** 2, h_lat, w_lat)
    # 2×2 patch pack: (B, 64, h_lat, w_lat) → (B, h_lat/2 * w_lat/2, 256)
    return pack_latents(mask)


def unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """(B, H/2*W/2, C*4) → (B, C, H, W)"""
    B, _, packed_dim = latents.shape
    C = packed_dim // 4
    h, w = height // 2, width // 2
    latents = latents.reshape(B, h, w, C, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5).reshape(B, C, height, width)
    return latents


def prepare_image_ids(height: int, width: int, device) -> torch.Tensor:
    """Positional IDs for FLUX's RoPE encoding. Returns (seq, 3) — no batch dim."""
    h, w = height // 16, width // 16  # after 8x VAE + 2x patch packing
    ids = torch.zeros(h, w, 3, device=device)
    ids[..., 1] = ids[..., 1] + torch.arange(h, device=device)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(w, device=device)[None, :]
    return ids.reshape(h * w, 3)


def get_sigmas(timesteps: torch.Tensor) -> torch.Tensor:
    """Flow matching: sigma_t = t (linear schedule)."""
    return timesteps.float()


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def train(cfg):
    proj_cfg = ProjectConfiguration(project_dir=cfg.output.dir, logging_dir=cfg.output.dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with="wandb",
        project_config=proj_cfg,
    )
    set_seed(42)

    os.makedirs(cfg.output.checkpoints, exist_ok=True)
    os.makedirs(cfg.output.dir, exist_ok=True)

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": {"name": cfg.logging.run_name}},
        )

    dtype = torch.bfloat16 if cfg.training.mixed_precision == "bf16" else torch.float32
    device = accelerator.device

    # -- Frozen backbone --
    logger.info("Loading frozen FLUX.1-Fill components …")
    vae, transformer = load_frozen_components(cfg.model.base_model, dtype, device)

    # -- Trainable ControlNet --
    logger.info("Initializing FluxControlNetModel …")
    controlnet_cfg = cfg.model.controlnet
    controlnet = FluxControlNetModel(
        num_layers=controlnet_cfg.num_layers,
        num_single_layers=controlnet_cfg.num_single_layers,
        in_channels=controlnet_cfg.in_channels,
        attention_head_dim=controlnet_cfg.attention_head_dim,
        num_attention_heads=controlnet_cfg.num_attention_heads,
        joint_attention_dim=controlnet_cfg.joint_attention_dim,
        pooled_projection_dim=controlnet_cfg.pooled_projection_dim,
        guidance_embeds=controlnet_cfg.guidance_embeds,
    )
    controlnet = controlnet.to(dtype=dtype)
    controlnet.train()

    # -- Pre-computed text embeddings (empty prompt) --
    if not os.path.exists(cfg.output.embeddings_cache):
        raise FileNotFoundError(
            f"Text embeddings not found at {cfg.output.embeddings_cache}. "
            "Run scripts/precompute_flux_embeddings.py first."
        )
    emb_cache = torch.load(cfg.output.embeddings_cache, map_location="cpu", weights_only=True)
    prompt_embeds        = emb_cache["prompt_embeds"].to(dtype=dtype)         # (1,512,4096)
    pooled_prompt_embeds = emb_cache["pooled_prompt_embeds"].to(dtype=dtype)  # (1,768)

    # -- Data --
    train_loader, val_loader = make_dataloaders(
        cfg.data.hdf5_path,
        image_size=cfg.training.image_size,
        batch_size=cfg.training.batch_size,
        val_split=cfg.data.val_split,
        num_workers=cfg.data.num_workers,
    )

    # -- Optimizer --
    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=cfg.training.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )
    total_steps = cfg.training.num_train_epochs * math.ceil(
        len(train_loader) / cfg.training.gradient_accumulation_steps
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # -- Accelerate prepare --
    controlnet, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        controlnet, optimizer, train_loader, val_loader, scheduler
    )

    H = W = cfg.training.image_size
    img_ids = prepare_image_ids(H, W, device)  # (seq, 3) — no batch dim

    global_step = 0
    for epoch in range(cfg.training.num_train_epochs):
        controlnet.train()
        pbar = tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}")

        for batch in pbar:
            with accelerator.accumulate(controlnet):

                # -- Encode images --
                image_latents        = encode_images(vae, batch["image"].to(dtype))
                masked_image_latents = encode_images(vae, batch["masked_image"].to(dtype))
                condition_latents    = encode_images(vae, batch["condition"].to(dtype))

                # Mask: pixel-fold + 2x2 pack → (B, seq, 256)
                # Mirrors FluxFillPipeline.prepare_mask_latents exactly.
                mask_packed = pack_mask(
                    batch["mask_binary"].to(dtype), vae_scale_factor=8
                ).to(device)

                B = image_latents.shape[0]

                # -- Flow matching noise --
                noise    = torch.randn_like(image_latents)
                u        = torch.normal(mean=0.0, std=1.0, size=(B,), device=device)
                t        = torch.sigmoid(u)              # logit-normal in (0,1)
                t_expand = t.view(B, 1, 1)
                noisy_latents = (1.0 - t_expand) * image_latents + t_expand * noise

                # FLUX.1-Fill hidden_states = cat([noisy, masked_image, mask])
                #   noisy:        (B, seq, 64)
                #   masked_image: (B, seq, 64)
                #   mask:         (B, seq, 256)  ← pixel-folded, NOT VAE-encoded
                #   total:        (B, seq, 384)  ← matches transformer in_channels=384
                noisy_model_input = torch.cat(
                    [noisy_latents, masked_image_latents, mask_packed], dim=-1
                )

                # -- Broadcast text embeddings to batch --
                pe  = prompt_embeds.expand(B, -1, -1).to(device)
                ppe = pooled_prompt_embeds.expand(B, -1).to(device)

                # -- ControlNet forward --
                # ControlNet sees only the noisy latents (64ch); masked-image
                # concatenation is handled inside the transformer.
                controlnet_block_samples, controlnet_single_block_samples = accelerator.unwrap_model(
                    controlnet
                )(
                    hidden_states=noisy_latents,        # (B, seq, 64)
                    controlnet_cond=condition_latents,  # (B, seq, 64)
                    conditioning_scale=cfg.training.controlnet_conditioning_scale,
                    timestep=t * 1000,                  # FLUX expects 0-1000 range
                    guidance=torch.full((B,), 3.5, device=device, dtype=dtype),
                    encoder_hidden_states=pe,
                    pooled_projections=ppe,
                    img_ids=img_ids,
                    txt_ids=torch.zeros(pe.shape[1], 3, device=device, dtype=dtype),
                    return_dict=False,
                )

                # -- Transformer forward --
                noise_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=t * 1000,
                    guidance=torch.full((B,), 3.5, device=device, dtype=dtype),
                    encoder_hidden_states=pe,
                    pooled_projections=ppe,
                    img_ids=img_ids,
                    txt_ids=torch.zeros(pe.shape[1], 3, device=device, dtype=dtype),
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                    return_dict=False,
                )[0]  # (B, seq, 64)

                # -- Flow matching loss: predict velocity v = eps - x_0 --
                target = noise - image_latents
                loss   = F.mse_loss(noise_pred.float(), target.float())

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), cfg.training.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if global_step % cfg.training.log_every == 0 and accelerator.is_main_process:
                accelerator.log({"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0]}, step=global_step)

        # -- Checkpoint --
        if (epoch + 1) % cfg.training.save_every == 0 and accelerator.is_main_process:
            ckpt_dir = f"{cfg.output.checkpoints}/epoch_{epoch+1:04d}"
            accelerator.unwrap_model(controlnet).save_pretrained(ckpt_dir)
            logger.info(f"Saved ControlNet → {ckpt_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_flux.yaml")
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    train(cfg)
