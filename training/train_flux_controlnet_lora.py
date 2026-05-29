"""
FLUX.1-Fill: train ControlNet + a LoRA adapter on the transformer.

Memory budget on A100 80 GB (bf16, no gradient checkpointing):
  - transformer base (bf16, frozen): ~24 GB
  - transformer LoRA params (trainable, tiny): a few hundred MB
  - ControlNet (bf16, trainable):  ~3 GB + grads + 8-bit AdamW state
  - VAE (bf16, frozen): ~0.5 GB
  - activations + buffers: rest of the budget

Compared to train_flux_controlnet_a100.py this adds LoRA on the transformer
attention modules so the denoiser itself can adapt to the hand-inpainting
distribution (the ControlNet alone can only inject residuals).

Usage:
    python training/train_flux_controlnet_lora.py --config configs/train_flux_a100_lora.yaml
"""

import argparse
import datetime as _datetime
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
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.hand_dataset import make_dataloaders

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_frozen_components(model_id: str, dtype: torch.dtype, device: str):
    """VAE (frozen) + transformer in bf16. LoRA is added in train()."""
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=dtype
    ).to(device)
    transformer = FluxTransformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=dtype
    ).to(device)
    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    return vae, transformer


def encode_images(vae, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.mode()
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pack_latents(latents)


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
    transformer.eval()
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

        masked_image_latents = encode_images(vae, masked_image)
        condition_latents    = encode_images(vae, condition)
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

            with torch.amp.autocast(device_type="cuda", dtype=dtype):
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
    transformer.train()
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
    # Per-run subdirectory so reruns don't overwrite previous results.
    run_id = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir     = os.path.join(cfg.output.dir, run_id)
    checkpoint_dir = os.path.join(cfg.output.checkpoints, run_id)
    run_name       = f"{cfg.logging.run_name}_{run_id}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    proj_cfg = ProjectConfiguration(project_dir=output_dir, logging_dir=output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with="wandb",
        project_config=proj_cfg,
    )
    set_seed(42)

    accelerator.init_trackers(
        project_name=cfg.logging.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        init_kwargs={"wandb": {"name": run_name}},
    )
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Output dir:     {output_dir}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")

    dtype = torch.bfloat16 if cfg.training.mixed_precision == "bf16" else torch.float32
    device = accelerator.device

    if not os.path.exists(cfg.data.hdf5_path):
        raise FileNotFoundError(f"Dataset not found at {cfg.data.hdf5_path}.")

    logger.info("Loading frozen FLUX.1-Fill components (bf16) …")
    vae, transformer = load_frozen_components(cfg.model.base_model, dtype, device)

    # --- LoRA on the transformer's attention modules ---
    lora_cfg = cfg.training.lora
    target_modules = list(lora_cfg.target_modules)
    logger.info(
        f"Adding LoRA (r={lora_cfg.rank}, alpha={lora_cfg.alpha}) to "
        f"transformer modules: {target_modules}"
    )
    transformer_lora_config = LoraConfig(
        r=lora_cfg.rank,
        lora_alpha=lora_cfg.alpha,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)
    # LoRA matrices land in fp32 by default; cast back to bf16 to match the
    # rest of the model.
    for name, p in transformer.named_parameters():
        if p.requires_grad:
            p.data = p.data.to(dtype=dtype)
    n_lora = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    logger.info(f"Trainable LoRA params on transformer: {n_lora/1e6:.2f}M")

    # --- ControlNet (fully trainable) ---
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
    controlnet.train()

    # Optional warm restart: load both ControlNet and LoRA weights from a
    # previous checkpoint dir produced by this script. We start a new run
    # (fresh global_step + fresh LR schedule); only the weights are reused.
    resume_from = getattr(cfg, "resume_from", None)
    if resume_from:
        if not os.path.isdir(resume_from):
            raise FileNotFoundError(f"--resume_from {resume_from} not found")
        logger.info(f"Resuming weights from {resume_from}")
        cn_loaded = FluxControlNetModel.from_pretrained(
            os.path.join(resume_from, "controlnet"), torch_dtype=dtype
        )
        controlnet.load_state_dict(cn_loaded.state_dict())
        del cn_loaded
        lora_state = torch.load(
            os.path.join(resume_from, "transformer_lora.pt"), map_location="cpu"
        )
        set_peft_model_state_dict(transformer, lora_state)
        logger.info("ControlNet + LoRA weights loaded")

    if not os.path.exists(cfg.output.embeddings_cache):
        raise FileNotFoundError(
            f"Text embeddings not found at {cfg.output.embeddings_cache}. "
            "Run scripts/precompute_flux_embeddings.py first."
        )
    emb_cache = torch.load(cfg.output.embeddings_cache, map_location="cpu", weights_only=True)
    prompt_embeds        = emb_cache["prompt_embeds"].to(dtype=dtype)
    pooled_prompt_embeds = emb_cache["pooled_prompt_embeds"].to(dtype=dtype)

    train_loader, val_loader = make_dataloaders(
        cfg.data.hdf5_path,
        image_size=cfg.training.image_size,
        batch_size=cfg.training.batch_size,
        val_split=cfg.data.val_split,
        num_workers=cfg.data.num_workers,
    )

    # Single optimizer for ControlNet + transformer-LoRA params.
    trainable_params = (
        list(controlnet.parameters())
        + [p for p in transformer.parameters() if p.requires_grad]
    )
    optimizer = bnb.optim.AdamW8bit(
        trainable_params,
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

    controlnet, transformer, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        controlnet, transformer, optimizer, train_loader, val_loader, scheduler
    )

    H = W = cfg.training.image_size
    img_ids = prepare_image_ids(H, W, device)

    global_step = 0

    logger.info("Running smoke-test inference at step 0 …")
    results = run_inference(
        transformer=accelerator.unwrap_model(transformer), vae=vae,
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
    accelerator.unwrap_model(transformer).train()

    for epoch in range(cfg.training.num_train_epochs):
        controlnet.train()
        accelerator.unwrap_model(transformer).train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            with accelerator.accumulate(controlnet, transformer):

                image_latents        = encode_images(vae, batch["image"].to(device, dtype=dtype))
                masked_image_latents = encode_images(vae, batch["masked_image"].to(device, dtype=dtype))
                condition_latents    = encode_images(vae, batch["condition"].to(device, dtype=dtype))
                mask_packed          = pack_mask(batch["mask_binary"].to(device, dtype=dtype), vae_scale_factor=8)

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

                with torch.amp.autocast(device_type="cuda", dtype=dtype):
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
                    accelerator.clip_grad_norm_(trainable_params, cfg.training.max_grad_norm)
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
                        ckpt_dir = f"{checkpoint_dir}/step_{global_step:06d}"
                        os.makedirs(ckpt_dir, exist_ok=True)
                        accelerator.unwrap_model(controlnet).save_pretrained(f"{ckpt_dir}/controlnet")
                        lora_state = get_peft_model_state_dict(accelerator.unwrap_model(transformer))
                        torch.save(lora_state, f"{ckpt_dir}/transformer_lora.pt")
                        logger.info(f"Saved ControlNet + LoRA → {ckpt_dir}")

                    if global_step % cfg.training.eval_every_iters == 0:
                        logger.info(f"Running inference at step {global_step} …")
                        results = run_inference(
                            transformer=accelerator.unwrap_model(transformer), vae=vae,
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
                        accelerator.unwrap_model(transformer).train()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    accelerator.end_training()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_flux_a100_lora.yaml")
    parser.add_argument("--resume_from", default=None,
                        help="Checkpoint dir with controlnet/ and transformer_lora.pt")
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    if args.resume_from:
        cfg.resume_from = args.resume_from
    train(cfg)
