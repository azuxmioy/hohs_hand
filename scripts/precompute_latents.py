"""
Pre-compute VAE latents for the hand dataset.

For each sample in the source HDF5, encodes
  - image (full)
  - masked_image (hand zeroed)
  - condition (skeleton-on-UV composite)
at the target resolution and writes a new HDF5 cache. The binary mask and
small raw RGB previews are also stored for visualization during inference.

Throughput-optimized version: CPU preprocessing runs in DataLoader workers
in parallel with VAE batched encoding (3 × batch images per VAE call).

Usage:
    python scripts/precompute_latents.py \\
        --src /data/hohs2/datasets/data.h5 \\
        --dst /data/hohs2/datasets/data_latents_512.h5 \\
        --resolution 512 --batch_size 8 --num_workers 4
"""
import argparse
import os

import h5py
import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class PrecomputeSource(Dataset):
    """CPU-side preprocessing: resize, compute mask, composite condition."""

    def __init__(self, src_path, resolution):
        self.src_path = src_path
        self.R = resolution
        with h5py.File(src_path, "r") as f:
            self.n = f["crops"].shape[0]
        self._f = None  # opened lazily per worker

    def __len__(self):
        return self.n

    def _file(self):
        if self._f is None:
            self._f = h5py.File(self.src_path, "r")
        return self._f

    def __getitem__(self, idx):
        f = self._file()
        crop     = f["crops"][idx]
        mask     = f["masks"][idx]
        skeleton = f["skeletons"][idx]
        uv       = f["uvs"][idx]

        R = self.R
        if crop.shape[0] != R:
            crop     = np.array(Image.fromarray(crop).resize((R, R), Image.BILINEAR))
            mask     = np.array(Image.fromarray(mask).resize((R, R), Image.NEAREST))
            skeleton = np.array(Image.fromarray(skeleton).resize((R, R), Image.BILINEAR))
            uv       = np.array(Image.fromarray(uv).resize((R, R), Image.BILINEAR))

        mb = (mask > 0).astype(np.uint8)
        masked = crop * (1 - mb[..., None])

        skel_f = skeleton.astype(np.float32) / 127.5 - 1.0
        uv_f   = uv.astype(np.float32) / 127.5 - 1.0
        skel_alpha = (skel_f.max(axis=-1) + 1.0) / 2.0
        cond_f = uv_f * (1 - skel_alpha[..., None]) + skel_f * skel_alpha[..., None]
        condition = ((cond_f + 1) * 127.5).clip(0, 255).astype(np.uint8)

        img_t  = torch.from_numpy(crop).permute(2, 0, 1).float() / 127.5 - 1.0
        mskd_t = torch.from_numpy(masked).permute(2, 0, 1).float() / 127.5 - 1.0
        cnd_t  = torch.from_numpy(condition).permute(2, 0, 1).float() / 127.5 - 1.0

        return {
            "idx":           idx,
            "img_t":         img_t,
            "mskd_t":        mskd_t,
            "cnd_t":         cnd_t,
            "mask_binary":   torch.from_numpy(mb),
            "image_rgb":     torch.from_numpy(crop),
            "masked_rgb":    torch.from_numpy(masked),
            "condition_rgb": torch.from_numpy(condition),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="/data/hohs2/datasets/data.h5")
    parser.add_argument("--dst", default="/data/hohs2/datasets/data_latents_512.h5")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.1-Fill-dev")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    R = args.resolution
    lat_dim = R // 8

    print(f"Loading VAE from {args.model_id} ...")
    vae = AutoencoderKL.from_pretrained(
        args.model_id, subfolder="vae", torch_dtype=dtype
    ).to(device)
    vae.eval()

    ds = PrecomputeSource(args.src, R)
    n = len(ds)
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f"Encoding {n} samples at {R}x{R} -> {args.dst}")
    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    dst = h5py.File(args.dst, "w")
    bs = args.batch_size
    dst.create_dataset("image_lat",     (n, 16, lat_dim, lat_dim), dtype="float16",
                       chunks=(bs, 16, lat_dim, lat_dim))
    dst.create_dataset("masked_lat",    (n, 16, lat_dim, lat_dim), dtype="float16",
                       chunks=(bs, 16, lat_dim, lat_dim))
    dst.create_dataset("condition_lat", (n, 16, lat_dim, lat_dim), dtype="float16",
                       chunks=(bs, 16, lat_dim, lat_dim))
    dst.create_dataset("mask_binary",   (n, R, R),    dtype="uint8", chunks=(bs, R, R))
    dst.create_dataset("image_rgb",     (n, R, R, 3), dtype="uint8", chunks=(bs, R, R, 3))
    dst.create_dataset("masked_rgb",    (n, R, R, 3), dtype="uint8", chunks=(bs, R, R, 3))
    dst.create_dataset("condition_rgb", (n, R, R, 3), dtype="uint8", chunks=(bs, R, R, 3))
    dst.attrs["resolution"] = R
    dst.attrs["vae_shift_factor"] = float(vae.config.shift_factor)
    dst.attrs["vae_scaling_factor"] = float(vae.config.scaling_factor)

    shift = vae.config.shift_factor
    scale = vae.config.scaling_factor

    with torch.no_grad():
        for batch in tqdm(dl):
            B = batch["img_t"].shape[0]
            # Stack 3 image variants into a single (3B,3,R,R) tensor for one VAE call
            stacked = torch.cat([batch["img_t"], batch["mskd_t"], batch["cnd_t"]], dim=0)
            stacked = stacked.to(device, dtype=dtype, non_blocking=True)

            lat = vae.encode(stacked).latent_dist.mode()
            lat = (lat - shift) * scale

            img_lat  = lat[0:B].float().cpu().numpy().astype(np.float16)
            mskd_lat = lat[B:2 * B].float().cpu().numpy().astype(np.float16)
            cnd_lat  = lat[2 * B:3 * B].float().cpu().numpy().astype(np.float16)

            idxs = batch["idx"].numpy()
            i0, i1 = int(idxs[0]), int(idxs[-1]) + 1   # shuffle=False ⇒ contiguous
            dst["image_lat"][i0:i1]     = img_lat
            dst["masked_lat"][i0:i1]    = mskd_lat
            dst["condition_lat"][i0:i1] = cnd_lat
            dst["mask_binary"][i0:i1]   = batch["mask_binary"].numpy()
            dst["image_rgb"][i0:i1]     = batch["image_rgb"].numpy()
            dst["masked_rgb"][i0:i1]    = batch["masked_rgb"].numpy()
            dst["condition_rgb"][i0:i1] = batch["condition_rgb"].numpy()

    dst.close()
    print("Done.")


if __name__ == "__main__":
    main()
