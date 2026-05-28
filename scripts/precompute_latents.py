"""
Pre-compute VAE latents for the hand dataset.

For each sample in the source HDF5, encodes
  - image (full)
  - masked_image (hand zeroed)
  - condition (skeleton-on-UV composite)
at the target resolution and writes a new HDF5 cache. The binary mask and
small raw RGB previews are also stored for visualization during inference.

Run once before training. Result lives outside the training loop so the
VAE is never touched during the training step.

Usage:
    python scripts/precompute_latents.py \\
        --src /data/hohs2/datasets/data.h5 \\
        --dst /data/hohs2/datasets/data_latents_512.h5 \\
        --resolution 512
"""
import argparse
import os

import h5py
import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="/data/hohs2/datasets/data.h5")
    parser.add_argument("--dst", default="/data/hohs2/datasets/data_latents_512.h5")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.1-Fill-dev")
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    R = args.resolution
    lat_dim = R // 8

    print(f"Loading VAE from {args.model_id} ...")
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae", torch_dtype=dtype).to(device)
    vae.eval()

    src = h5py.File(args.src, "r")
    n = src["crops"].shape[0]
    print(f"Encoding {n} samples at {R}x{R} -> {args.dst}")

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    dst = h5py.File(args.dst, "w")
    dst.create_dataset("image_lat",     (n, 16, lat_dim, lat_dim), dtype="float16")
    dst.create_dataset("masked_lat",    (n, 16, lat_dim, lat_dim), dtype="float16")
    dst.create_dataset("condition_lat", (n, 16, lat_dim, lat_dim), dtype="float16")
    dst.create_dataset("mask_binary",   (n, R, R), dtype="uint8")
    dst.create_dataset("image_rgb",     (n, R, R, 3), dtype="uint8")
    dst.create_dataset("masked_rgb",    (n, R, R, 3), dtype="uint8")
    dst.create_dataset("condition_rgb", (n, R, R, 3), dtype="uint8")
    dst.attrs["resolution"] = R
    dst.attrs["vae_shift_factor"] = float(vae.config.shift_factor)
    dst.attrs["vae_scaling_factor"] = float(vae.config.scaling_factor)

    shift = vae.config.shift_factor
    scale = vae.config.scaling_factor

    for i in tqdm(range(n)):
        crop     = src["crops"][i]      # uint8 (H, W, 3)
        mask     = src["masks"][i]      # uint8 (H, W)
        skeleton = src["skeletons"][i]
        uv       = src["uvs"][i]

        # Resize to target R (BILINEAR for RGB, NEAREST for mask)
        if crop.shape[0] != R:
            crop     = np.array(Image.fromarray(crop).resize((R, R), Image.BILINEAR))
            mask     = np.array(Image.fromarray(mask).resize((R, R), Image.NEAREST))
            skeleton = np.array(Image.fromarray(skeleton).resize((R, R), Image.BILINEAR))
            uv       = np.array(Image.fromarray(uv).resize((R, R), Image.BILINEAR))

        mb = (mask > 0).astype(np.uint8)              # (R, R) binary
        masked = crop * (1 - mb[..., None])           # (R, R, 3)

        # Skeleton-on-UV composite (matches data/hand_dataset.py)
        skel_f = skeleton.astype(np.float32) / 127.5 - 1.0
        uv_f   = uv.astype(np.float32) / 127.5 - 1.0
        skel_alpha = (skel_f.max(axis=-1) + 1.0) / 2.0
        cond_f = uv_f * (1 - skel_alpha[..., None]) + skel_f * skel_alpha[..., None]
        condition = ((cond_f + 1) * 127.5).clip(0, 255).astype(np.uint8)

        # Encode 3 images in one VAE forward
        img_t  = torch.from_numpy(crop).permute(2, 0, 1).float() / 127.5 - 1.0
        mskd_t = torch.from_numpy(masked).permute(2, 0, 1).float() / 127.5 - 1.0
        cnd_t  = torch.from_numpy(condition).permute(2, 0, 1).float() / 127.5 - 1.0
        batch = torch.stack([img_t, mskd_t, cnd_t], dim=0).to(device, dtype=dtype)

        with torch.no_grad():
            lat = vae.encode(batch).latent_dist.mode()        # (3, 16, lat_dim, lat_dim)
            lat = (lat - shift) * scale

        dst["image_lat"][i]     = lat[0].float().cpu().numpy().astype(np.float16)
        dst["masked_lat"][i]    = lat[1].float().cpu().numpy().astype(np.float16)
        dst["condition_lat"][i] = lat[2].float().cpu().numpy().astype(np.float16)
        dst["mask_binary"][i]   = mb
        dst["image_rgb"][i]     = crop
        dst["masked_rgb"][i]    = masked
        dst["condition_rgb"][i] = condition

    src.close()
    dst.close()
    print(f"Done.")


if __name__ == "__main__":
    main()
