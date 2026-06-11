"""Refine the hand masks in a conditioning HDF5 with SAM2 (occlusion-aware).

The MANO mesh-silhouette mask covers the full hand projection, including pixels
where an object is actually in front of the hand. SAM2, prompted by our reliable
21 projected hand keypoints, segments only the *visible* hand, so object-occluded
regions are excluded -- closer to the real hand segmentation used in training.

Reads an h5 from arctic_make_conditions.py (needs crops + keypoints_2d_output),
writes a copy with masks replaced by SAM2 masks. Runs in the `hohs_hand` env.

    python inference/sam2_masks.py \
        --in-h5  /data/hohs2/arctic/box_grab_01_ego_s3.h5 \
        --out-h5 /data/hohs2/arctic/box_grab_01_ego_s3_sam.h5 \
        --preview /data/hohs2/arctic/sam2_mask_preview.png
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_fill_holes


def make_disk(r):
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-h5", required=True)
    ap.add_argument("--out-h5", required=True)
    ap.add_argument("--model", default="facebook/sam2.1-hiera-large")
    ap.add_argument("--dilate", type=int, default=6,
                    help="px disk dilation of the final hand mask")
    ap.add_argument("--mesh-pad", type=int, default=12,
                    help="px the mesh silhouette is dilated before intersecting with SAM2 "
                         "(bounds SAM2 to the hand region; removes forearm/table over-seg)")
    ap.add_argument("--min-area", type=int, default=200,
                    help="fallback to the mesh mask if final area < this")
    ap.add_argument("--preview", default=None)
    ap.add_argument("--preview-n", type=int, default=6)
    args = ap.parse_args()

    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fin = h5py.File(args.in_h5, "r")
    N = fin["crops"].shape[0]
    S = fin["crops"].shape[1]
    crops = fin["crops"]
    kpts = fin["keypoints_2d_output"]          # (N,21,2) in crop px
    mesh_masks = fin["masks"]                   # (N,S,S) 0/255

    out_masks = np.zeros((N, S, S), dtype=np.uint8)
    disk = make_disk(args.dilate) if args.dilate > 0 else None
    n_fallback = 0

    for i in range(N):
        img = np.ascontiguousarray(crops[i])    # (S,S,3) uint8 RGB
        pts = np.asarray(kpts[i], dtype=np.float32)
        # keep only in-bounds keypoints as positive prompts
        inb = (pts[:, 0] >= 0) & (pts[:, 0] < S) & (pts[:, 1] >= 0) & (pts[:, 1] < S)
        pts = pts[inb]
        mesh_i = mesh_masks[i] > 127
        kp = np.asarray(kpts[i], dtype=np.float32)
        # Single positive prompt at the WRIST (kp 0) -- it sits on the hand, away from
        # the fingertips/palm where the object is grasped, so the prompt never lands on
        # the object. Fall back to the centroid of in-bounds keypoints if the wrist is
        # out of frame.
        wrist = kp[0]
        if 0 <= wrist[0] < S and 0 <= wrist[1] < S:
            prompt = wrist[None]
        elif len(pts) >= 1:
            prompt = pts.mean(0, keepdims=True)
        else:
            out_masks[i] = mesh_masks[i]; n_fallback += 1; continue
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            predictor.set_image(img)
            masks, scores, _ = predictor.predict(
                point_coords=prompt, point_labels=np.ones((1,), np.int32),
                multimask_output=True)          # 3 masks: ~finger / hand / hand+arm
        # pick the scale that best matches the projected-hand region (mesh) -> the
        # hand-scale mask, not the whole-arm one.
        mesh_d = binary_dilation(mesh_i, structure=make_disk(args.mesh_pad)) if args.mesh_pad > 0 else mesh_i
        ious = [float((m.astype(bool) & mesh_d).sum()) / (float((m.astype(bool) | mesh_d).sum()) + 1e-6)
                for m in masks]
        best = masks[int(np.argmax(ious))].astype(bool)
        final = best & mesh_d                    # clip any forearm spillover to hand region
        if final.sum() < args.min_area:
            out_masks[i] = mesh_masks[i]; n_fallback += 1; continue
        final = binary_fill_holes(final)
        if disk is not None:
            final = binary_dilation(final, structure=disk)
        out_masks[i] = (final * 255).astype(np.uint8)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N}")

    print(f"SAM2 masks done; fell back to mesh mask on {n_fallback}/{N}")

    # write copy with masks replaced
    Path(args.out_h5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out_h5, "w") as fout:
        for k in fin.keys():
            if k == "masks":
                fout.create_dataset("masks", data=out_masks, dtype=np.uint8, compression="lzf")
            else:
                fin.copy(k, fout)
        for ak, av in fin.attrs.items():
            fout.attrs[ak] = av
    print(f"Wrote {args.out_h5}")

    if args.preview:
        from PIL import Image
        rows = []
        for i in range(min(args.preview_n, N)):
            crop = crops[i]
            mesh = np.stack([mesh_masks[i]] * 3, -1)
            sam = np.stack([out_masks[i]] * 3, -1)
            mb = (out_masks[i] > 127)[..., None]
            masked = (crop * (1 - mb)).astype(np.uint8)
            rows.append(np.concatenate([crop, mesh, sam, masked], axis=1))
        Image.fromarray(np.concatenate(rows, 0)).save(args.preview)
        print(f"Preview -> {args.preview}  (cols: crop | mesh_mask | sam2_mask | masked_input)")


if __name__ == "__main__":
    main()
