"""Build hohs_hand conditioning (crop/skeleton/UV/mask) for PALM samples.

PALM ships per subject: `cameras.npy` (7 cams, each K/dist/Rt/h/w), `poses.npy`
(MANO: betas, global_orient, hand_pose, transl), and images/masks/mano per camera.
This reproduces the same conditioning recipe as the ARCTIC pipeline (reusing its
render helpers) but with PALM's loader front-end. PALM is right-hand only.

Run in the `hohs_hand` env:
    python inference/palm_make_conditions.py \
        --subj-dir /data/hohs2/palm/_peek/0000 \
        --mano-dir /data/hohs2/datasets/arctic_dl/mano_v1_2/models \
        --uv-right /data/hohs2/arctic/MANO_UV_right.obj \
        --gesture 000001 --scale 3 --mask-dilate 8 \
        --out /data/hohs2/palm/palm_0000_g1.h5 --debug
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.arctic_make_conditions import (
    build_mano, mano_forward, parse_mano_uv_obj, keypoint_crop_box,
    project_to_image, transform_world2cam, render_crop, render_skeleton,
    render_uv, render_mask, KEYPOINT_NAMES,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subj-dir", required=True, help="unzipped PALM subject dir (has cameras.npy, poses.npy, images/)")
    ap.add_argument("--mano-dir", required=True)
    ap.add_argument("--uv-right", required=True)
    ap.add_argument("--gesture", default="000001")
    ap.add_argument("--cameras", default="MCU_01,MCU_02,MCU_03,MCU_04,MCU_05,MCU_06,MCU_07")
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--out-size", type=int, default=512)
    ap.add_argument("--mask-dilate", type=int, default=8)
    ap.add_argument("--flat-hand", action="store_true",
                    help="use flat_hand_mean=True for the MANO layer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug", action="store_true", help="save a kp-on-image overlay for the first cam")
    args = ap.parse_args()

    subj = Path(args.subj_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cams = np.load(subj / "cameras.npy", allow_pickle=True).item()["cameras"]
    poses = np.load(subj / "poses.npy", allow_pickle=True).item()
    g = int(args.gesture) - 1     # frame 000001 -> index 0

    layer = build_mano(args.mano_dir, is_rhand=True, device=device)
    vt, ft = parse_mano_uv_obj(Path(args.uv_right))
    faces = layer.faces.astype(np.int64)

    rot   = np.asarray(poses["global_orient"])[g:g + 1]
    pose  = np.asarray(poses["hand_pose"])[g:g + 1]
    trans = np.asarray(poses["transl"])[g:g + 1]
    betas = np.asarray(poses["betas"])[:1]
    verts_w, joints_w = mano_forward(layer, rot, pose, trans, betas, device)
    verts_w, joints_w = verts_w[0], joints_w[0]

    acc = {k: [] for k in ["crops", "masks", "skeletons", "uvs", "kp2d", "kp3d",
                           "is_right", "frame_keys", "side", "view"]}
    for cam_id in args.cameras.split(","):
        cam = cams[cam_id]
        Rt = np.asarray(cam["Rt"], dtype=np.float32)
        K = np.asarray(cam["K"], dtype=np.float32)
        H, W = int(cam["height"]), int(cam["width"])
        img_path = subj / "images" / cam_id / f"{args.gesture}.jpg"
        if not img_path.exists():
            print(f"  [skip] missing {img_path}"); continue
        frame = np.array(Image.open(img_path).convert("RGB"))

        verts_cam = transform_world2cam(verts_w, Rt)
        joints_cam = transform_world2cam(joints_w, Rt)
        kp2d = project_to_image(joints_cam, K)

        if args.debug and not acc["crops"]:
            im = Image.open(img_path).convert("RGB"); d = ImageDraw.Draw(im)
            for x, y in kp2d:
                d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=(255, 0, 0))
            im.resize((W // 4, H // 4)).save(str(Path(args.out).with_suffix("")) + f"_dbg_{cam_id}.png")
            print(f"  debug overlay saved for {cam_id}; kp2d x[{kp2d[:,0].min():.0f},{kp2d[:,0].max():.0f}] "
                  f"y[{kp2d[:,1].min():.0f},{kp2d[:,1].max():.0f}] (img {W}x{H})")

        x1, y1, x2, y2 = keypoint_crop_box(kp2d, args.scale, (H, W))
        cw, ch = x2 - x1, y2 - y1
        if cw < 4 or ch < 4:
            print(f"  [skip] {cam_id}: degenerate crop"); continue
        sx, sy = args.out_size / cw, args.out_size / ch
        kp2d_out = (kp2d - np.array([x1, y1])) * np.array([sx, sy])

        crop = render_crop(frame, y1, y2, x1, x2, args.out_size)
        skel = render_skeleton(kp2d_out, args.out_size)
        uv = render_uv(verts_cam, faces, vt, ft, K, x1, y1, cw, ch, args.out_size)
        mask = render_mask(verts_cam, faces, K, x1, y1, cw, ch, args.out_size, dilate=args.mask_dilate)

        acc["crops"].append(crop); acc["masks"].append(mask)
        acc["skeletons"].append(skel); acc["uvs"].append(uv)
        acc["kp2d"].append(kp2d_out.astype(np.float32)); acc["kp3d"].append(joints_cam.astype(np.float32))
        acc["is_right"].append(1); acc["frame_keys"].append(f"{subj.name}/{cam_id}/{args.gesture}")
        acc["side"].append("right"); acc["view"].append(cam_id)
        print(f"  {cam_id} done")

    n = len(acc["crops"])
    if n == 0:
        raise SystemExit("no samples produced")
    import h5py, json
    dt = h5py.special_dtype(vlen=str)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("crops", data=np.stack(acc["crops"]), dtype=np.uint8, compression="lzf")
        f.create_dataset("masks", data=np.stack(acc["masks"]), dtype=np.uint8, compression="lzf")
        f.create_dataset("skeletons", data=np.stack(acc["skeletons"]), dtype=np.uint8, compression="lzf")
        f.create_dataset("uvs", data=np.stack(acc["uvs"]), dtype=np.uint8, compression="lzf")
        f.create_dataset("keypoints_2d_output", data=np.stack(acc["kp2d"]), dtype=np.float32)
        f.create_dataset("keypoints_3d", data=np.stack(acc["kp3d"]), dtype=np.float32)
        f.create_dataset("is_right", data=np.array(acc["is_right"], np.uint8))
        f.create_dataset("frame_keys", data=np.array(acc["frame_keys"], dtype=object), dtype=dt)
        f.create_dataset("side", data=np.array(acc["side"], dtype=object), dtype=dt)
        f.create_dataset("view", data=np.array(acc["view"], dtype=object), dtype=dt)
        f.attrs["keypoint_names"] = json.dumps(KEYPOINT_NAMES)
        f.attrs["output_size"] = args.out_size
    print(f"Wrote {args.out}  ({n} samples)")


if __name__ == "__main__":
    main()
