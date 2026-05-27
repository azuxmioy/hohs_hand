"""HDF5 hand dataset for diffusion model training."""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image
import random


KEYPOINT_CONNECTIONS = [
    # thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
]

FINGER_COLORS = [
    (255, 0, 0),    # thumb - red
    (255, 165, 0),  # index - orange
    (255, 255, 0),  # middle - yellow
    (0, 255, 0),    # ring - green
    (0, 0, 255),    # pinky - blue
]


def render_skeleton(kp2d, size=512, line_width=3, point_radius=5):
    """Render 21 keypoints as a skeleton image (H, W, 3) uint8."""
    from PIL import ImageDraw
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for finger_idx, connections in enumerate([
        [(0,1),(1,2),(2,3),(3,4)],
        [(0,5),(5,6),(6,7),(7,8)],
        [(0,9),(9,10),(10,11),(11,12)],
        [(0,13),(13,14),(14,15),(15,16)],
        [(0,17),(17,18),(18,19),(19,20)],
    ]):
        color = FINGER_COLORS[finger_idx]
        for a, b in connections:
            x0, y0 = float(kp2d[a, 0]), float(kp2d[a, 1])
            x1, y1 = float(kp2d[b, 0]), float(kp2d[b, 1])
            draw.line([(x0, y0), (x1, y1)], fill=color, width=line_width)
    for j in range(21):
        x, y = float(kp2d[j, 0]), float(kp2d[j, 1])
        r = point_radius
        draw.ellipse([(x-r, y-r), (x+r, y+r)], fill=(255, 255, 255))
    return np.array(img)


class HandDataset(Dataset):
    """
    Loads hand data from an HDF5 file.

    Each sample contains:
      image    : (3, H, W) float32 in [-1, 1]  — RGB crop
      mask     : (1, H, W) float32 in [-1, 1]  — hand segmentation
      skeleton : (3, H, W) float32 in [-1, 1]  — skeleton rendering
      uv       : (3, H, W) float32 in [-1, 1]  — UV texture map
      keypoints_2d : (21, 2) float32 normalized to [0, 1]
      keypoints_3d : (21, 3) float32 in original metric units
      is_right : bool
    """

    def __init__(
        self,
        hdf5_path,
        image_size=256,
        use_stored_skeleton=True,
        augment=True,
        indices=None,
    ):
        self.hdf5_path = hdf5_path
        self.image_size = image_size
        self.use_stored_skeleton = use_stored_skeleton
        self.augment = augment

        with h5py.File(hdf5_path, "r") as f:
            self.length = f["crops"].shape[0]
            if indices is None:
                self.indices = list(range(self.length))
            else:
                self.indices = list(indices)

        self.color_jitter = T.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
        )

    def __len__(self):
        return len(self.indices)

    def _load(self, idx):
        with h5py.File(self.hdf5_path, "r") as f:
            crop     = f["crops"][idx]          # (512, 512, 3) uint8
            mask     = f["masks"][idx]           # (512, 512)    uint8
            skeleton = f["skeletons"][idx]       # (512, 512, 3) uint8
            uv       = f["uvs"][idx]             # (512, 512, 3) uint8
            kp2d     = f["keypoints_2d_output"][idx].copy()  # (21, 2) float32
            kp3d     = f["keypoints_3d"][idx].copy()         # (21, 3) float32
            is_right = bool(f["is_right"][idx])
        return crop, mask, skeleton, uv, kp2d, kp3d, is_right

    def _to_pil(self, arr):
        return Image.fromarray(arr)

    def __getitem__(self, item):
        idx = self.indices[item]
        crop, mask, skeleton, uv, kp2d, kp3d, is_right = self._load(idx)

        # Optionally re-render skeleton from keypoints (identical result but
        # allows consistent rendering after any future augmentation)
        if not self.use_stored_skeleton:
            skeleton = render_skeleton(kp2d, size=crop.shape[0])

        # PIL images for transform pipeline
        img_pil  = self._to_pil(crop)
        mask_pil = self._to_pil(mask)
        skel_pil = self._to_pil(skeleton)
        uv_pil   = self._to_pil(uv)

        # -- Augmentation --
        if self.augment:
            # Random 90° rotation (0, 90, 180, 270) — no reflection to
            # preserve left/right hand consistency.
            rot_k = random.randint(0, 3)  # number of 90° CCW rotations
            if rot_k > 0:
                S = img_pil.width  # square image
                img_pil  = TF.rotate(img_pil,  rot_k * 90, interpolation=TF.InterpolationMode.BILINEAR)
                mask_pil = TF.rotate(mask_pil, rot_k * 90, interpolation=TF.InterpolationMode.NEAREST)
                skel_pil = TF.rotate(skel_pil, rot_k * 90, interpolation=TF.InterpolationMode.BILINEAR)
                uv_pil   = TF.rotate(uv_pil,   rot_k * 90, interpolation=TF.InterpolationMode.BILINEAR)
                kp2d = kp2d.copy()
                cx = cy = (S - 1) / 2.0
                for _ in range(rot_k):
                    new_x = cy - (kp2d[:, 1] - cy)
                    new_y = cx + (kp2d[:, 0] - cx)
                    kp2d[:, 0], kp2d[:, 1] = new_x, new_y

            # Random crop / resize
            i, j, h, w = T.RandomResizedCrop.get_params(
                img_pil, scale=(0.8, 1.0), ratio=(0.9, 1.1)
            )
            img_pil  = TF.resized_crop(img_pil,  i, j, h, w, (self.image_size, self.image_size), TF.InterpolationMode.BILINEAR)
            mask_pil = TF.resized_crop(mask_pil, i, j, h, w, (self.image_size, self.image_size), TF.InterpolationMode.NEAREST)
            skel_pil = TF.resized_crop(skel_pil, i, j, h, w, (self.image_size, self.image_size), TF.InterpolationMode.BILINEAR)
            uv_pil   = TF.resized_crop(uv_pil,   i, j, h, w, (self.image_size, self.image_size), TF.InterpolationMode.BILINEAR)
            # adjust keypoints
            orig_size = crop.shape[0]
            kp2d = kp2d.copy()
            kp2d[:, 0] = (kp2d[:, 0] - j) / w * self.image_size
            kp2d[:, 1] = (kp2d[:, 1] - i) / h * self.image_size

            # Color jitter on RGB only
            img_pil = self.color_jitter(img_pil)

        else:
            img_pil  = TF.resize(img_pil,  (self.image_size, self.image_size), TF.InterpolationMode.BILINEAR)
            mask_pil = TF.resize(mask_pil, (self.image_size, self.image_size), TF.InterpolationMode.NEAREST)
            skel_pil = TF.resize(skel_pil, (self.image_size, self.image_size), TF.InterpolationMode.BILINEAR)
            uv_pil   = TF.resize(uv_pil,   (self.image_size, self.image_size), TF.InterpolationMode.BILINEAR)
            scale = self.image_size / crop.shape[0]
            kp2d = kp2d * scale

        # -- To tensor, normalize to [-1, 1] --
        to_t = lambda pil: TF.to_tensor(pil) * 2.0 - 1.0
        image_t    = to_t(img_pil)                         # (3, H, W)
        mask_t     = TF.to_tensor(mask_pil) * 2.0 - 1.0   # (1, H, W)
        skeleton_t = to_t(skel_pil)                        # (3, H, W)
        uv_t       = to_t(uv_pil)                          # (3, H, W)

        # Inpainting inputs -----------------------------------------------
        # mask_binary: 1 where hand is (region to inpaint), 0 elsewhere
        mask_binary = (mask_t > 0).float()                 # (1, H, W)
        masked_image_t = image_t * (1.0 - mask_binary)    # (3, H, W) hand zeroed out

        # ControlNet condition: skeleton lines overlaid on UV map.
        # Skeleton has black background; wherever it is bright, use skeleton
        # colour; elsewhere show UV. Keeps both pose and texture info in RGB.
        skel_alpha  = (skeleton_t.max(dim=0, keepdim=True).values + 1.0) / 2.0  # [0,1]
        condition_t = uv_t * (1.0 - skel_alpha) + skeleton_t * skel_alpha       # (3, H, W)

        kp2d_norm = torch.from_numpy(kp2d / self.image_size).float()  # (21,2) in [0,1]
        kp3d_t    = torch.from_numpy(kp3d).float()                    # (21,3)

        return {
            "image":         image_t,          # (3,H,W) target full image
            "masked_image":  masked_image_t,   # (3,H,W) image with hand removed
            "mask":          mask_t,           # (1,H,W) hand region, [-1,1]
            "mask_binary":   mask_binary,      # (1,H,W) hand region, {0,1}
            "condition":     condition_t,      # (3,H,W) skeleton-on-UV composite
            "skeleton":      skeleton_t,       # (3,H,W) raw skeleton
            "uv":            uv_t,             # (3,H,W) raw UV map
            "keypoints_2d":  kp2d_norm,        # (21,2) in [0,1]
            "keypoints_3d":  kp3d_t,           # (21,3) metric
            "is_right":      torch.tensor(is_right, dtype=torch.bool),
        }


def make_dataloaders(hdf5_path, image_size=256, batch_size=8,
                     val_split=0.1, num_workers=4, seed=42):
    """Return (train_loader, val_loader)."""
    rng = np.random.default_rng(seed)
    with h5py.File(hdf5_path, "r") as f:
        n = f["crops"].shape[0]
    idx = rng.permutation(n)
    split = max(1, int(n * val_split))
    val_idx, train_idx = idx[:split], idx[split:]

    train_ds = HandDataset(hdf5_path, image_size=image_size, augment=True,  indices=train_idx)
    val_ds   = HandDataset(hdf5_path, image_size=image_size, augment=False, indices=val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
