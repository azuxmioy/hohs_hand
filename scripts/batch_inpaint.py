"""Batch-inpaint all samples from a conditioning HDF5 and save individual images.

Loads the FLUX.1-Fill + ControlNet + LoRA checkpoint once, then iterates through
every sample in the HDF5, runs the denoising loop, and saves each generated hand
image as a separate PNG.  Produces a metadata JSON that maps sample indices to
output image paths (consumed by h5_to_hamer_npz.py).

Usage:
    python scripts/batch_inpaint.py \
        --config configs/train_flux_a100_lora_calib.yaml \
        --checkpoint /data/hohs2/checkpoints/flux_controlnet_lora/<run>/step_XXXXXX \
        --h5 /data/hohs2/palm/palm_0000_g1.h5 \
        --embeddings /data/hohs2/outputs/flux_controlnet/text_embeddings.pt \
        --out-dir /data/hohs2/palm/inpainted_0000_g1 \
        --num-steps 30 --guidance 30.0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.hand_dataset import HandDataset
from inference.arctic_inpaint import (
    encode_images,
    pack_mask,
    prepare_image_ids,
    unpack_latents,
)


def to_uint8(t):
    return ((t * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


@torch.no_grad()
def inpaint_all(transformer, vae, controlnet, loader,
                prompt_embeds, pooled_prompt_embeds,
                device, dtype, image_size, out_dir,
                num_steps=30, guidance_scale=30.0, seed=0):
    controlnet.eval()
    transformer.eval()
    img_ids = prepare_image_ids(image_size, image_size, device)
    txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=dtype)
    timesteps = torch.linspace(1.0, 1.0 / num_steps, num_steps, device=device, dtype=dtype)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for idx, batch in enumerate(loader):
        B = 1
        masked_image = batch["masked_image"][:B].to(device, dtype=dtype)
        condition = batch["condition"][:B].to(device, dtype=dtype)
        mask_binary = batch["mask_binary"][:B].to(device, dtype=dtype)

        masked_image_latents = encode_images(vae, masked_image)
        condition_latents = encode_images(vae, condition)
        mask_packed = pack_mask(mask_binary, vae_scale_factor=8)

        gen = torch.Generator(device=device).manual_seed(seed + idx)
        seq_len = (image_size // 16) ** 2
        latents = torch.randn(B, seq_len, 64, device=device, dtype=dtype, generator=gen)
        pe = prompt_embeds.expand(B, -1, -1).to(device)
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

        fname = f"{idx:05d}.png"
        Image.fromarray(to_uint8(generated[0].cpu().float())).save(out_dir / fname)
        manifest.append({"index": idx, "image": fname})
        print(f"  [{idx + 1}/{len(loader)}] saved {fname}")

    meta_path = out_dir / "manifest.json"
    with open(meta_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Done: {len(manifest)} images -> {out_dir}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_flux_a100_lora_calib.yaml")
    ap.add_argument("--checkpoint", required=True,
                    help="step_XXXXXX dir with controlnet/ + transformer_lora.pt")
    ap.add_argument("--h5", required=True, help="conditioning HDF5 from palm_make_conditions.py")
    ap.add_argument("--embeddings", required=True, help="text_embeddings.pt (empty-prompt cache)")
    ap.add_argument("--out-dir", required=True, help="output directory for generated images")
    ap.add_argument("--num-steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=30.0)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from diffusers import AutoencoderKL, FluxControlNetModel, FluxTransformer2DModel
    from omegaconf import OmegaConf
    from peft import LoraConfig
    from peft.utils import set_peft_model_state_dict

    cfg = OmegaConf.load(args.config)
    dtype = torch.bfloat16
    device = "cuda"

    print("Loading VAE + transformer (bf16) ...")
    vae = AutoencoderKL.from_pretrained(
        cfg.model.base_model, subfolder="vae", torch_dtype=dtype).to(device)
    transformer = FluxTransformer2DModel.from_pretrained(
        cfg.model.base_model, subfolder="transformer", torch_dtype=dtype).to(device)
    vae.requires_grad_(False)
    transformer.requires_grad_(False)

    print("Attaching LoRA adapter ...")
    lc = cfg.training.lora
    transformer.add_adapter(LoraConfig(
        r=lc.rank, lora_alpha=lc.alpha, init_lora_weights="gaussian",
        target_modules=list(lc.target_modules)))
    lora_state = torch.load(
        Path(args.checkpoint) / "transformer_lora.pt", map_location="cpu")
    set_peft_model_state_dict(transformer, lora_state)
    transformer.to(dtype=dtype)

    print("Loading ControlNet ...")
    controlnet = FluxControlNetModel.from_pretrained(
        Path(args.checkpoint) / "controlnet", torch_dtype=dtype).to(device)

    emb = torch.load(args.embeddings, map_location="cpu", weights_only=True)
    prompt_embeds = emb["prompt_embeds"].to(dtype=dtype)
    pooled_prompt_embeds = emb["pooled_prompt_embeds"].to(dtype=dtype)

    ds = HandDataset(args.h5, image_size=args.image_size, augment=False,
                     use_stored_skeleton=True, mask_dilation_max=0)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"Dataset: {len(ds)} samples")

    inpaint_all(
        transformer, vae, controlnet, loader,
        prompt_embeds, pooled_prompt_embeds,
        device, dtype, args.image_size, args.out_dir,
        num_steps=args.num_steps, guidance_scale=args.guidance, seed=args.seed)


if __name__ == "__main__":
    main()
