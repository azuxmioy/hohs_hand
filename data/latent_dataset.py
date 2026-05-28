"""HDF5-cached latent dataset with 90° rotation augmentation.

Reads the cache produced by scripts/precompute_latents.py. Returns
pre-encoded VAE latents plus the binary mask and small RGB previews
(for inference visualization). Random 90° rotations are applied in
latent space — FLUX's VAE is convolutional so rot90 commutes well.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class LatentHandDataset(Dataset):
    """
    Each sample contains:
      image_lat    : (16, h, w) float32 — VAE latents of the full image
      masked_lat   : (16, h, w) float32 — VAE latents of the masked image
      condition_lat: (16, h, w) float32 — VAE latents of skeleton-on-UV
      mask_binary  : (1, H, W)  float32 — binary mask in {0,1} at image res
      image        : (3, H, W)  float32 in [-1, 1] — for visualization
      masked_image : (3, H, W)  float32 in [-1, 1]
      condition    : (3, H, W)  float32 in [-1, 1]
    """

    def __init__(self, h5_path, augment=True, indices=None):
        self.h5_path = h5_path
        self.augment = augment
        with h5py.File(h5_path, "r") as f:
            n = f["image_lat"].shape[0]
        self.indices = list(range(n)) if indices is None else list(indices)
        self._f = None  # opened lazily per worker

    def __len__(self):
        return len(self.indices)

    def _file(self):
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r")
        return self._f

    def __getitem__(self, item):
        idx = self.indices[item]
        f = self._file()

        image_lat     = torch.from_numpy(f["image_lat"][idx].astype(np.float32))
        masked_lat    = torch.from_numpy(f["masked_lat"][idx].astype(np.float32))
        condition_lat = torch.from_numpy(f["condition_lat"][idx].astype(np.float32))
        mask_binary   = torch.from_numpy(f["mask_binary"][idx]).float().unsqueeze(0)
        image_rgb     = torch.from_numpy(f["image_rgb"][idx]).float().permute(2, 0, 1) / 127.5 - 1.0
        masked_rgb    = torch.from_numpy(f["masked_rgb"][idx]).float().permute(2, 0, 1) / 127.5 - 1.0
        condition_rgb = torch.from_numpy(f["condition_rgb"][idx]).float().permute(2, 0, 1) / 127.5 - 1.0

        if self.augment:
            k = int(np.random.randint(0, 4))
            if k > 0:
                image_lat     = torch.rot90(image_lat,     k, dims=(-2, -1))
                masked_lat    = torch.rot90(masked_lat,    k, dims=(-2, -1))
                condition_lat = torch.rot90(condition_lat, k, dims=(-2, -1))
                mask_binary   = torch.rot90(mask_binary,   k, dims=(-2, -1))
                image_rgb     = torch.rot90(image_rgb,     k, dims=(-2, -1))
                masked_rgb    = torch.rot90(masked_rgb,    k, dims=(-2, -1))
                condition_rgb = torch.rot90(condition_rgb, k, dims=(-2, -1))

        return {
            "image_lat":     image_lat,
            "masked_lat":    masked_lat,
            "condition_lat": condition_lat,
            "mask_binary":   mask_binary,
            "image":         image_rgb,
            "masked_image":  masked_rgb,
            "condition":     condition_rgb,
        }


def make_latent_dataloaders(
    h5_path, batch_size=1, val_split=0.1, num_workers=4, seed=42
):
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, "r") as f:
        n = f["image_lat"].shape[0]
    idx = rng.permutation(n)
    split = max(1, int(n * val_split))
    val_idx, train_idx = idx[:split], idx[split:]

    train_ds = LatentHandDataset(h5_path, augment=True,  indices=train_idx)
    val_ds   = LatentHandDataset(h5_path, augment=False, indices=val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
