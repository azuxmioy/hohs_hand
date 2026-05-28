"""
FLUX.1-Fill + ControlNet training tuned for a single A100 (80 GB).

Differences from train_flux_controlnet_latent.py:
  - bf16 transformer instead of NF4 (no quantization overhead)
  - Gradient checkpointing OFF (faster forward/backward)
  - Single-GPU only — no DDP, no NCCL barriers
  - Larger micro-batch (batch_size > 1 OK because activation memory fits)

Pre-requisite: run scripts/precompute_latents.py to build the latent cache.

Usage:
    python training/train_flux_controlnet_a100.py --config configs/train_flux_a100.yaml
"""

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, FluxControlNetModel, FluxTransformer2DModel
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.latent_dataset import make_latent_dataloaders

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_frozen_components(model_id: str, dtype: torch.dtype, device: str):
    """VAE + transformer in bf16 (no quantization). Both frozen."""
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=dtype
    ).to(device)

    transformer = FluxTransformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=dtype
    ).to(device)

    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    # Gradient checkpointing intentionally OFF — A100 has the memory.
    return vae, transformer


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    B, C, H, W = latents.shape
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5).reshape(B, (H // 2) * (W // 2), C * 4)
    return latents


def pack_mask(mask: torch.Tensor, vae_scale_factor: int = 8) -> torch.Tensor:
    B, _, H, W = mask.shape
    mask = mask[:, 0]
    h_lat = H // vae_scale_factor
    w_lat = W // vae_scale_factor
    mask = mask.view(B, h_lat, vae_scale_factor, w_lat, vae_scale_factor)
    mask = mask.permute(0, 2, 4, 1, 3).reshape(B, vae_scale_factor ** 2, h_lat, w_lat)
    return pack_latents(mask)


def unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    B, _, packed_dim = latents.shape
    C = packed_dim // 4
    h, w = height // 2, width // 2
    latents = latents.reshape(B, h, w, C, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5).reshape(B, C, height, width)
    return latents


def prepare_image_ids(height: int, width: int, device) -> torch.Tensor:
    h, w = height // 16, width // 16
    ids = torch.zeros(h, w, 3, device=device)
    ids[..., 1] = ids[..., 1] + torch.arange(h, device=device)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(w, device=device)[None, :]
    return ids.reshape(h * w, 3)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    transformer, vae, controlnet, val_loader,
    prompt_embeds, pooled_prompt_embeds,
    device, dtype, image_size,
    num_steps=30, num_samples=4, guidance_scale=30.0,
):
    controlnet.eval()
    results = []

    img_ids = prepare_image_ids(image_size, image_size, device)
    txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=dtype)
    timesteps = torch.linspace(1.0, 1.0 / num_steps, num_steps, device=device, dtype=dtype)

    for batch in val_loader:
        if len(results) >= num_samples:
            break

        B = 1
        image        = batch["image"][:B].to(device, dtype=dtype)
        masked_image = batch["masked_image"][:B].to(device, dtype=dtype)
        condition    = batch["condition"][:B].to(device, dtype=dtype)
        mask_binary  = batch["mask_binary"][:B].to(device, dtype=dtype)

        masked_image_latents = pack_latents(batch["masked_lat"][:B].to(device, dtype=dtype))
        condition_latents    = pack_latents(batch["condition_lat"][:B].to(device, dtype=dtype))
        mask_packed          = pack_mask(mask_binary, vae_scale_factor=8)

        seq_len = (image_size // 16) ** 2
        latents = torch.randn(B, seq_len, 64, device=device, dtype=dtype)

        pe  = prompt_embeds.expand(B, -1, -1).to(device)
        ppe = pooled_prompt_embeds.expand(B, -1).to(device)
        guidance = torch.full((B,), guidance_scale, device=device, dtype=dtype)

        for t_val in timesteps:
            t_batch = torch.full((B,), t_val.item(), device=device, dtype=dtype)
            noisy_model_input = torch.cat(
                [latents, masked_image_latents, mask_packed], dim=-1
            )

            cn_block, cn_single = controlnet(
                hidden_states=latents,
                controlnet_cond=condition_latents,
                conditioning_scale=1.0,
                timestep=t_batch,
                guidance=guidance,
                encoder_hidden_states=pe,
                pooled_projections=ppe,
                img_ids=img_ids,
                txt_ids=txt_ids,
                return_dict=False,
            )
            v_pred = transformer(
                hidden_states=noisy_model_input,
                timestep=t_batch,
                guidance=guidance,
                encoder_hidden_states=pe,
                pooled_projections=ppe,
                img_ids=img_ids,
                txt_ids=txt_ids,
                controlnet_block_samples=cn_block,
                controlnet_single_block_samples=cn_single,
                return_dict=False,
            )[0]

            latents = latents - (1.0 / num_steps) * v_pred

        lat_4d = unpack_latents(latents, image_size // 8, image_size // 8)
        lat_4d = lat_4d / vae.config.scaling_factor + vae.config.shift_factor
        generated = vae.decode(lat_4d.to(dtype)).sample.clamp(-1, 1)

        results.append({
            "original":  image[0].cpu().float(),
            "masked":    masked_image[0].cpu().float(),
            "condition": condition[0].cpu().float(),
            "generated": generated[0].cpu().float(),
        })

    controlnet.train()
    return results


def make_image_grid(results):
    import numpy as np
    rows = []
    for r in results:
        row = torch.stack([r["original"], r["masked"], r["condition"], r["generated"]], dim=0)
        row = (row * 0.5 + 0.5).clamp(0, 1)
        rows.append(row)
    grid = torch.cat(rows, dim=0)
    return (grid.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)


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

    accelerator.init_trackers(
        project_name=cfg.logging.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        init_kwargs={"wandb": {"name": cfg.logging.run_name}},
    )

    dtype = torch.bfloat16 if cfg.training.mixed_precision == "bf16" else torch.float32
    device = accelerator.device

    if not os.path.exists(cfg.data.latent_cache):
        raise FileNotFoundError(
            f"Latent cache not found at {cfg.data.latent_cache}. "
            "Run scripts/precompute_latents.py first."
        )

    logger.info("Loading frozen FLUX.1-Fill components (bf16, no NF4) …")
    vae, transformer = load_frozen_components(cfg.model.base_model, dtype, device)

    logger.info("Initializing FluxControlNetModel …")
    cnc = cfg.model.controlnet
    controlnet = FluxControlNetModel(
        num_layers=cnc.num_layers,
        num_single_layers=cnc.num_single_layers,
        in_channels=cnc.in_channels,
        attention_head_dim=cnc.attention_head_dim,
        num_attention_heads=cnc.num_attention_heads,
        joint_attention_dim=cnc.joint_attention_dim,
        pooled_projection_dim=cnc.pooled_projection_dim,
        guidance_embeds=cnc.guidance_embeds,
    ).to(dtype=dtype)
    # Gradient checkpointing intentionally OFF for speed.
    controlnet.train()

    if not os.path.exists(cfg.output.embeddings_cache):
        raise FileNotFoundError(
            f"Text embeddings not found at {cfg.output.embeddings_cache}. "
            "Run scripts/precompute_flux_embeddings.py first."
        )
    emb_cache = torch.load(cfg.output.embeddings_cache, map_location="cpu", weights_only=True)
    prompt_embeds        = emb_cache["prompt_embeds"].to(dtype=dtype)
    pooled_prompt_embeds = emb_cache["pooled_prompt_embeds"].to(dtype=dtype)

    train_loader, val_loader = make_latent_dataloaders(
        cfg.data.latent_cache,
        batch_size=cfg.training.batch_size,
        val_split=cfg.data.val_split,
        num_workers=cfg.data.num_workers,
    )

    optimizer = bnb.optim.AdamW8bit(
        controlnet.parameters(),
        lr=cfg.training.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )
    total_steps = cfg.training.num_train_epochs * math.ceil(
        len(train_loader) / cfg.training.gradient_accumulation_steps
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=total_steps,
    )

    controlnet, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        controlnet, optimizer, train_loader, val_loader, scheduler
    )

    H = W = cfg.training.image_size
    img_ids = prepare_image_ids(H, W, device)

    global_step = 0

    logger.info("Running smoke-test inference at step 0 …")
    results = run_inference(
        transformer=transformer, vae=vae,
        controlnet=accelerator.unwrap_model(controlnet),
        val_loader=val_loader,
        prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
        device=device, dtype=dtype, image_size=cfg.training.image_size,
        num_steps=cfg.training.num_inference_steps,
        num_samples=cfg.training.num_eval_samples,
    )
    if results:
        grid_np = make_image_grid(results)
        wandb.log(
            {"val/samples": [wandb.Image(grid_np[i]) for i in range(len(grid_np))]},
            step=0,
        )
    controlnet.train()

    for epoch in range(cfg.training.num_train_epochs):
        controlnet.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            with accelerator.accumulate(controlnet):

                image_latents        = pack_latents(batch["image_lat"].to(dtype))
                masked_image_latents = pack_latents(batch["masked_lat"].to(dtype))
                condition_latents    = pack_latents(batch["condition_lat"].to(dtype))
                mask_packed          = pack_mask(batch["mask_binary"].to(dtype), vae_scale_factor=8)

                B = image_latents.shape[0]

                noise    = torch.randn_like(image_latents)
                u        = torch.normal(mean=0.0, std=1.0, size=(B,), device=device)
                t        = torch.sigmoid(u)
                t_expand = t.view(B, 1, 1)
                noisy_latents = (1.0 - t_expand) * image_latents + t_expand * noise

                noisy_model_input = torch.cat(
                    [noisy_latents, masked_image_latents, mask_packed], dim=-1
                )

                pe  = prompt_embeds.expand(B, -1, -1).to(device)
                ppe = pooled_prompt_embeds.expand(B, -1).to(device)

                controlnet_block_samples, controlnet_single_block_samples = controlnet(
                    hidden_states=noisy_latents,
                    controlnet_cond=condition_latents,
                    conditioning_scale=cfg.training.controlnet_conditioning_scale,
                    timestep=t,
                    guidance=torch.full((B,), 30.0, device=device, dtype=dtype),
                    encoder_hidden_states=pe,
                    pooled_projections=ppe,
                    img_ids=img_ids,
                    txt_ids=torch.zeros(pe.shape[1], 3, device=device, dtype=dtype),
                    return_dict=False,
                )

                noise_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=t,
                    guidance=torch.full((B,), 30.0, device=device, dtype=dtype),
                    encoder_hidden_states=pe,
                    pooled_projections=ppe,
                    img_ids=img_ids,
                    txt_ids=torch.zeros(pe.shape[1], 3, device=device, dtype=dtype),
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                    return_dict=False,
                )[0]

                target = noise - image_latents
                loss   = F.mse_loss(noise_pred.float(), target.float())

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), cfg.training.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % cfg.training.log_every == 0:
                        accelerator.log(
                            {"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0]},
                            step=global_step,
                        )

                    if global_step % cfg.training.save_every_iters == 0:
                        ckpt_dir = f"{cfg.output.checkpoints}/step_{global_step:06d}"
                        accelerator.unwrap_model(controlnet).save_pretrained(ckpt_dir)
                        logger.info(f"Saved ControlNet → {ckpt_dir}")

                    if global_step % cfg.training.eval_every_iters == 0:
                        logger.info(f"Running inference at step {global_step} …")
                        results = run_inference(
                            transformer=transformer, vae=vae,
                            controlnet=accelerator.unwrap_model(controlnet),
                            val_loader=val_loader,
                            prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            device=device, dtype=dtype,
                            image_size=cfg.training.image_size,
                            num_steps=cfg.training.num_inference_steps,
                            num_samples=cfg.training.num_eval_samples,
                        )
                        if results:
                            grid_np = make_image_grid(results)
                            wandb.log(
                                {"val/samples": [wandb.Image(grid_np[i]) for i in range(len(grid_np))]},
                                step=global_step,
                            )
                        controlnet.train()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    accelerator.end_training()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_flux_a100.yaml")
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    train(cfg)
