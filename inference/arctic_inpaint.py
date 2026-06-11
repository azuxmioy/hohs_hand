"""Inpaint ARCTIC hand crops with the trained FLUX.1-Fill + ControlNet + LoRA checkpoint.

Consumes an HDF5 produced by arctic_make_conditions.py (same keys as data.h5) and
reuses data/hand_dataset.py:HandDataset(augment=False) so preprocessing is identical
to training. The denoise loop / latent packing are copied verbatim from
training/train_flux_controlnet_lora.py:run_inference (kept self-contained to avoid
importing wandb/accelerate).

Run with the `hohs_hand` env:
    python inference/arctic_inpaint.py \
        --config configs/train_flux_a100_lora.yaml \
        --checkpoint /data/hohs2/checkpoints/flux_controlnet_lora/20260529_211730/step_008000 \
        --h5 /data/hohs2/arctic/box_grab_01_v1.h5 \
        --embeddings /data/hohs2/outputs/flux_controlnet/text_embeddings.pt \
        --out-dir /data/hohs2/arctic/results_box_grab_01_v1 \
        --num-samples 8 --num-steps 30 --guidance 30.0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL, FluxControlNetModel, FluxTransformer2DModel
from omegaconf import OmegaConf
from peft import LoraConfig
from peft.utils import set_peft_model_state_dict
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.hand_dataset import HandDataset


# --------------------------------------------------------------------------
# Latent packing helpers (verbatim from train_flux_controlnet_lora.py)
# --------------------------------------------------------------------------
def pack_latents(latents):
    B, C, H, W = latents.shape
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)
    return latents.permute(0, 2, 4, 1, 3, 5).reshape(B, (H // 2) * (W // 2), C * 4)


def encode_images(vae, images):
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.mode()
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pack_latents(latents)


def pack_mask(mask, vae_scale_factor=8):
    B, _, H, W = mask.shape
    mask = mask[:, 0]
    h_lat, w_lat = H // vae_scale_factor, W // vae_scale_factor
    mask = mask.view(B, h_lat, vae_scale_factor, w_lat, vae_scale_factor)
    mask = mask.permute(0, 2, 4, 1, 3).reshape(B, vae_scale_factor ** 2, h_lat, w_lat)
    return pack_latents(mask)


def unpack_latents(latents, height, width):
    B, _, packed_dim = latents.shape
    C = packed_dim // 4
    h, w = height // 2, width // 2
    latents = latents.reshape(B, h, w, C, 2, 2)
    return latents.permute(0, 3, 1, 4, 2, 5).reshape(B, C, height, width)


def prepare_image_ids(height, width, device):
    h, w = height // 16, width // 16
    ids = torch.zeros(h, w, 3, device=device)
    ids[..., 1] = ids[..., 1] + torch.arange(h, device=device)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(w, device=device)[None, :]
    return ids.reshape(h * w, 3)


@torch.no_grad()
def run_inference(transformer, vae, controlnet, loader,
                  prompt_embeds, pooled_prompt_embeds,
                  device, dtype, image_size,
                  num_steps=30, num_samples=8, guidance_scale=30.0, seed=0):
    controlnet.eval(); transformer.eval()
    results = []
    img_ids = prepare_image_ids(image_size, image_size, device)
    txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=dtype)
    timesteps = torch.linspace(1.0, 1.0 / num_steps, num_steps, device=device, dtype=dtype)

    for batch in loader:
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
            noisy_model_input = torch.cat([latents, masked_image_latents, mask_packed], dim=-1)
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                cn_block, cn_single = controlnet(
                    hidden_states=latents, controlnet_cond=condition_latents,
                    conditioning_scale=1.0, timestep=t_batch, guidance=guidance,
                    encoder_hidden_states=pe, pooled_projections=ppe,
                    img_ids=img_ids, txt_ids=txt_ids, return_dict=False)
                v_pred = transformer(
                    hidden_states=noisy_model_input, timestep=t_batch, guidance=guidance,
                    encoder_hidden_states=pe, pooled_projections=ppe,
                    img_ids=img_ids, txt_ids=txt_ids,
                    controlnet_block_samples=cn_block,
                    controlnet_single_block_samples=cn_single, return_dict=False)[0]
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
    return results


def to_uint8(t):
    return ((t * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_flux_a100_lora.yaml")
    ap.add_argument("--checkpoint", required=True, help="step_XXXXXX dir with controlnet/ + transformer_lora.pt")
    ap.add_argument("--h5", required=True)
    ap.add_argument("--embeddings", required=True, help="text_embeddings.pt (empty-prompt cache)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--num-steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=30.0)
    ap.add_argument("--image-size", type=int, default=512)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    dtype = torch.bfloat16
    device = "cuda"

    print("Loading VAE + transformer (bf16) …")
    vae = AutoencoderKL.from_pretrained(cfg.model.base_model, subfolder="vae", torch_dtype=dtype).to(device)
    transformer = FluxTransformer2DModel.from_pretrained(
        cfg.model.base_model, subfolder="transformer", torch_dtype=dtype).to(device)
    vae.requires_grad_(False); transformer.requires_grad_(False)

    print("Attaching LoRA adapter and loading trained LoRA weights …")
    lc = cfg.training.lora
    transformer.add_adapter(LoraConfig(
        r=lc.rank, lora_alpha=lc.alpha, init_lora_weights="gaussian",
        target_modules=list(lc.target_modules)))
    lora_state = torch.load(Path(args.checkpoint) / "transformer_lora.pt", map_location="cpu")
    set_peft_model_state_dict(transformer, lora_state)
    transformer.to(dtype=dtype)

    print("Loading ControlNet from checkpoint …")
    controlnet = FluxControlNetModel.from_pretrained(
        Path(args.checkpoint) / "controlnet", torch_dtype=dtype).to(device)

    emb = torch.load(args.embeddings, map_location="cpu", weights_only=True)
    prompt_embeds = emb["prompt_embeds"].to(dtype=dtype)
    pooled_prompt_embeds = emb["pooled_prompt_embeds"].to(dtype=dtype)

    ds = HandDataset(args.h5, image_size=args.image_size, augment=False,
                     use_stored_skeleton=True, mask_dilation_max=0)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"Dataset: {len(ds)} samples")

    results = run_inference(
        transformer, vae, controlnet, loader,
        prompt_embeds, pooled_prompt_embeds, device, dtype, args.image_size,
        num_steps=args.num_steps, num_samples=args.num_samples, guidance_scale=args.guidance)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cols = ["original", "masked", "condition", "generated"]
    rows = []
    for i, r in enumerate(results):
        panels = [to_uint8(r[c]) for c in cols]
        row = np.concatenate(panels, axis=1)
        Image.fromarray(row).save(out / f"sample_{i:03d}.png")
        rows.append(row)
    if rows:
        Image.fromarray(np.concatenate(rows, axis=0)).save(out / "grid.png")
    print(f"Wrote {len(results)} samples + grid.png to {out}  (cols: {cols})")


if __name__ == "__main__":
    main()
