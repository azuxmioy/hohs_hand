"""Convert a PALM conditioning HDF5 + inpainted images into a HaMeR-compatible NPZ.

Reads the conditioning HDF5 (produced by palm_make_conditions.py with MANO param
fields) and maps each sample to its corresponding inpainted image.  Outputs a
compressed NPZ that can be loaded directly by HaMeR's ImageDataset for fine-tuning.

Usage:
    python scripts/h5_to_hamer_npz.py \
        --h5 /data/hohs2/palm/palm_0000_g1.h5 \
        --images /data/hohs2/palm/inpainted_0000_g1 \
        --output /data/hohs2/palm/train_hamer.npz \
        --image-root /data/hohs2/palm
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, nargs="+",
                    help="conditioning HDF5(s) with MANO params")
    ap.add_argument("--images", required=True, nargs="+",
                    help="directory(ies) with inpainted PNGs (1:1 with --h5)")
    ap.add_argument("--output", required=True, help="output HaMeR NPZ path")
    ap.add_argument("--image-root", default=None,
                    help="base dir to make image paths relative to")
    ap.add_argument("--bbox-padding", type=float, default=1.0,
                    help="padding factor applied to scale (default 1.0 = tight)")
    args = ap.parse_args()

    if len(args.h5) != len(args.images):
        raise ValueError("--h5 and --images must have the same number of entries")

    imgnames = []
    centers = []
    scales = []
    kp2d_all = []
    kp3d_all = []
    hand_poses = []
    has_hand_pose_all = []
    betas_all = []
    has_betas_all = []
    right_all = []
    extra_info_all = []
    image_root = Path(args.image_root).resolve() if args.image_root else None

    for h5_path, img_dir in zip(args.h5, args.images):
        img_dir = Path(img_dir)
        manifest_path = img_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            idx_to_fname = {e["index"]: e["image"] for e in manifest}
        else:
            idx_to_fname = None

        with h5py.File(h5_path, "r") as f:
            n = f["crops"].shape[0]
            kp2d = f["keypoints_2d_output"][:].astype(np.float32)  # (N, 21, 2)
            kp3d = f["keypoints_3d"][:].astype(np.float32)          # (N, 21, 3)
            is_right = f["is_right"][:].astype(np.float32)           # (N,)
            out_size = int(f.attrs.get("output_size", 512))

            has_mano = "global_orient" in f
            if has_mano:
                g_orient = f["global_orient"][:].astype(np.float32)  # (N, 3)
                h_pose = f["hand_pose"][:].astype(np.float32)        # (N, 45)
                betas = f["betas"][:].astype(np.float32)              # (N, 10)
            else:
                g_orient = np.zeros((n, 3), dtype=np.float32)
                h_pose = np.zeros((n, 45), dtype=np.float32)
                betas = np.zeros((n, 10), dtype=np.float32)

            frame_keys = [fk.decode() if isinstance(fk, bytes) else str(fk)
                          for fk in f["frame_keys"][:]]

        for i in range(n):
            if idx_to_fname is not None:
                fname = idx_to_fname.get(i, f"{i:05d}.png")
            else:
                fname = f"{i:05d}.png"
            img_path = img_dir / fname
            if not img_path.exists():
                print(f"  [skip] missing {img_path}")
                continue

            abs_path = img_path.resolve()
            if image_root:
                try:
                    rel = str(abs_path.relative_to(image_root))
                except ValueError:
                    rel = str(abs_path)
            else:
                rel = str(abs_path)

            imgnames.append(rel)

            center = np.array([out_size / 2.0, out_size / 2.0], dtype=np.float32)
            scale = np.array([out_size, out_size], dtype=np.float32) * args.bbox_padding
            centers.append(center)
            scales.append(scale)

            kp2d_conf = np.ones((21, 1), dtype=np.float32)
            kp2d_all.append(np.concatenate([kp2d[i], kp2d_conf], axis=1))  # (21, 3)

            kp3d_conf = np.ones((21, 1), dtype=np.float32)
            kp3d_all.append(np.concatenate([kp3d[i], kp3d_conf], axis=1))  # (21, 4)

            pose_48 = np.concatenate([g_orient[i], h_pose[i]])  # (48,)
            hand_poses.append(pose_48)
            has_hand_pose_all.append(1.0 if has_mano else 0.0)

            betas_10 = betas[i]
            if betas_10.shape[0] < 10:
                betas_10 = np.pad(betas_10, (0, 10 - betas_10.shape[0]))
            betas_all.append(betas_10[:10])
            has_betas_all.append(1.0 if has_mano else 0.0)

            right_all.append(float(is_right[i]))
            extra_info_all.append({
                "source_index": int(i),
                "source_h5": str(h5_path),
                "frame_key": frame_keys[i],
            })

    if not imgnames:
        raise SystemExit("No samples produced")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        imgname=np.asarray(imgnames, dtype=object),
        center=np.asarray(centers, dtype=np.float32),
        scale=np.asarray(scales, dtype=np.float32),
        hand_keypoints_2d=np.asarray(kp2d_all, dtype=np.float32),
        hand_keypoints_3d=np.asarray(kp3d_all, dtype=np.float32),
        hand_pose=np.asarray(hand_poses, dtype=np.float32),
        has_hand_pose=np.asarray(has_hand_pose_all, dtype=np.float32),
        betas=np.asarray(betas_all, dtype=np.float32),
        has_betas=np.asarray(has_betas_all, dtype=np.float32),
        right=np.asarray(right_all, dtype=np.float32),
        extra_info=np.asarray(extra_info_all, dtype=object),
    )
    print(f"Wrote {len(imgnames)} samples to {out_path}")


if __name__ == "__main__":
    main()
