"""Build hohs_hand conditioning (crop / mask / skeleton / UV) for an ARCTIC sequence.

ARCTIC ships ground-truth MANO params (world coord) + camera calibration, so we
skip HaMeR/stereo and reproduce the SAME conditioning recipe as the training
data-prep (generate_hand_crops.py):

  - square crop box from projected 2D keypoints (x`scale`)
  - 21-keypoint skeleton render
  - MANO mesh projected + UV-rasterized (u->R, v->G) using MANO_UV_{left,right}.obj
  - hand mask = filled silhouette of the projected MANO mesh (ARCTIC has no tight
    per-pixel hand mask for full images, so the mesh silhouette is the inpaint hole)
  - condition (built later in the dataset) = skeleton drawn over the UV map

Output is an HDF5 whose keys MATCH data.h5 exactly, so inference can reuse
data/hand_dataset.py:HandDataset(augment=False) for identical preprocessing.

Run with the `phd`/`hohs_hand` env (needs smplx + chumpy, patched for numpy>=1.24):
    python inference/arctic_make_conditions.py \
        --arctic-root /data/hohs2/datasets/arctic/data \
        --mano-dir   /data/hohs2/datasets/arctic/data/body_models/mano \
        --uv-left  /data/hohs2/arctic/MANO_UV_left.obj \
        --uv-right /data/hohs2/arctic/MANO_UV_right.obj \
        --subject s01 --seq box_grab_01 --view 1 \
        --stride 5 --max-frames 40 \
        --out /data/hohs2/arctic/box_grab_01_v1.h5
"""

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

# --------------------------------------------------------------------------
# chumpy / numpy-alias patch (mirrors phd/phd/fitter/pt/common.py) so the
# legacy MANO .pkl (which pickles chumpy arrays) unpickles under modern numpy.
# --------------------------------------------------------------------------
@contextlib.contextmanager
def monkey_patched_for_chumpy():
    import numpy as _np
    for name in ["bool", "int", "object", "str"]:
        if name not in dir(_np):
            try:
                sys.modules[f"numpy.{name}"] = getattr(_np, name + "_")
            except Exception:
                pass
    sys.modules["numpy.float"] = float
    sys.modules["numpy.complex"] = _np.complex128
    sys.modules["numpy.NINF"] = -_np.inf
    _np.NINF = -_np.inf
    _np.complex = _np.complex128
    _np.float = float
    if "unicode" not in dir(_np):
        sys.modules["numpy.unicode"] = _np.str_
    import inspect
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    yield


# --------------------------------------------------------------------------
# Skeleton / colours (verbatim from generate_hand_crops.py for parity)
# --------------------------------------------------------------------------
SKELETON_CONNECTIONS = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]
_C = {
    "wrist":  (255, 255, 255),
    "thumb":  (255,  80,  80),
    "index":  ( 80, 255,  80),
    "middle": ( 80,  80, 255),
    "ring":   (255, 255,  80),
    "pinky":  (255,  80, 255),
}
KP_COLORS = [
    _C["wrist"],
    *[_C["thumb"]] * 4, *[_C["index"]] * 4,
    *[_C["middle"]] * 4, *[_C["ring"]] * 4, *[_C["pinky"]] * 4,
]
CONN_COLORS = [
    _C["thumb"], _C["index"], _C["middle"], _C["ring"], _C["pinky"],
    *[_C["thumb"]] * 3, *[_C["index"]] * 3, *[_C["middle"]] * 3,
    *[_C["ring"]] * 3, *[_C["pinky"]] * 3,
]
KEYPOINT_NAMES = [
    "wrist",
    "thumb_mcp", "thumb_pip", "thumb_dip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

# smplx.MANO returns joints in MANO-native order
#   [wrist, index(1-3), middle(4-6), pinky(7-9), ring(10-12), thumb(13-15),
#    thumb_tip(16), index_tip(17), middle_tip(18), ring_tip(19), pinky_tip(20)]
# The training data (via HaMeR) used the "openpose" order
#   [wrist, thumb(4), index(4), middle(4), ring(4), pinky(4)]  == KEYPOINT_NAMES.
# HaMeR's mano_wrapper applies exactly this map; we replicate it so the rendered
# skeleton's finger chains + colours match what the ControlNet was trained on.
MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


# --------------------------------------------------------------------------
# OBJ UV parsing (verbatim)
# --------------------------------------------------------------------------
def parse_mano_uv_obj(obj_path: Path):
    vt, ft = [], []
    with open(obj_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("vt "):
                p = line.split()
                vt.append((float(p[1]), float(p[2])))
            elif line.startswith("f "):
                triplets = line.split()[1:]
                ft.append([int(t.split("/")[1]) - 1 for t in triplets])
    return np.array(vt, dtype=np.float32), np.array(ft, dtype=np.int32)


# --------------------------------------------------------------------------
# Geometry / renderers (ported from generate_hand_crops.py)
# --------------------------------------------------------------------------
def keypoint_crop_box(kp2d, scale, img_hw):
    H, W = img_hw
    cx = (kp2d[:, 0].min() + kp2d[:, 0].max()) / 2.0
    cy = (kp2d[:, 1].min() + kp2d[:, 1].max()) / 2.0
    half = max(kp2d[:, 0].max() - kp2d[:, 0].min(),
               kp2d[:, 1].max() - kp2d[:, 1].min()) * scale / 2.0
    x1 = max(0, int(round(cx - half)))
    y1 = max(0, int(round(cy - half)))
    x2 = min(W, int(round(cx + half)))
    y2 = min(H, int(round(cy + half)))
    return x1, y1, x2, y2


def project_to_image(verts_cam, K):
    x = K[0, 0] * verts_cam[:, 0] / verts_cam[:, 2] + K[0, 2]
    y = K[1, 1] * verts_cam[:, 1] / verts_cam[:, 2] + K[1, 2]
    return np.stack([x, y], axis=1)


def render_crop(frame_rgb, y1, y2, x1, x2, out_size):
    arr = frame_rgb[y1:y2, x1:x2]
    return np.array(Image.fromarray(arr).resize((out_size, out_size), Image.LANCZOS))


def render_skeleton(kp2d, out_size):
    img = Image.new("RGB", (out_size, out_size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    lw = max(2, out_size // 150)
    for idx, (i, j) in enumerate(SKELETON_CONNECTIONS):
        draw.line([tuple(kp2d[i].tolist()), tuple(kp2d[j].tolist())],
                  fill=CONN_COLORS[idx], width=lw)
    r = max(3, out_size // 100)
    for idx, pt in enumerate(kp2d):
        x, y = float(pt[0]), float(pt[1])
        draw.ellipse([x - r, y - r, x + r, y + r], fill=KP_COLORS[idx])
    return np.array(img)


def _rasterize(verts_cam, faces, K, x1, y1, cw, ch, out_size, shade_uv=None):
    """Painter's-algorithm rasterizer. If shade_uv=(vt,ft) returns UV image
    (u->R, v->G); else returns a white silhouette mask. (out_size,out_size,3)/(.,.) uint8."""
    verts_2d_full = project_to_image(verts_cam, K)
    verts_2d = (verts_2d_full - np.array([x1, y1], dtype=np.float32)) \
        * np.array([out_size / cw, out_size / ch], dtype=np.float32)
    order = np.argsort(verts_cam[:, 2][faces].mean(axis=1))[::-1]  # back to front

    if shade_uv is not None:
        vt, ft = shade_uv
        img = np.zeros((out_size, out_size, 3), dtype=np.uint8)
    else:
        img = np.zeros((out_size, out_size), dtype=np.uint8)

    for fi in order:
        f = faces[fi]
        p = verts_2d[f]
        x0, y0 = p[0, 0], p[0, 1]
        x1f, y1f = p[1, 0], p[1, 1]
        x2f, y2f = p[2, 0], p[2, 1]
        bx0 = max(0, int(np.floor(min(x0, x1f, x2f))))
        bx1 = min(out_size - 1, int(np.ceil(max(x0, x1f, x2f))))
        by0 = max(0, int(np.floor(min(y0, y1f, y2f))))
        by1 = min(out_size - 1, int(np.ceil(max(y0, y1f, y2f))))
        if bx1 < bx0 or by1 < by0:
            continue
        xs = np.arange(bx0, bx1 + 1, dtype=np.float32) + 0.5
        ys = np.arange(by0, by1 + 1, dtype=np.float32) + 0.5
        px, py = np.meshgrid(xs, ys)
        px, py = px.ravel(), py.ravel()
        denom = (y1f - y2f) * (x0 - x2f) + (x2f - x1f) * (y0 - y2f)
        if abs(denom) < 1e-8:
            continue
        w0 = ((y1f - y2f) * (px - x2f) + (x2f - x1f) * (py - y2f)) / denom
        w1 = ((y2f - y0) * (px - x2f) + (x0 - x2f) * (py - y2f)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
        if not inside.any():
            continue
        ix = px[inside].astype(np.int32)
        iy = py[inside].astype(np.int32)
        if shade_uv is not None:
            uv = vt[ft[fi]]
            u = (w0[inside] * uv[0, 0] + w1[inside] * uv[1, 0] + w2[inside] * uv[2, 0]).clip(0, 1)
            v = (w0[inside] * uv[0, 1] + w1[inside] * uv[1, 1] + w2[inside] * uv[2, 1]).clip(0, 1)
            img[iy, ix, 0] = (u * 255).astype(np.uint8)
            img[iy, ix, 1] = (v * 255).astype(np.uint8)
        else:
            img[iy, ix] = 255
    return img


def render_uv(verts_cam, faces, vt, ft, K, x1, y1, cw, ch, out_size):
    return _rasterize(verts_cam, faces, K, x1, y1, cw, ch, out_size, shade_uv=(vt, ft))


def render_mask(verts_cam, faces, K, x1, y1, cw, ch, out_size, dilate=24):
    """Filled MANO-mesh silhouette grown by a disk of radius `dilate` px (at out_size).
    The bare mesh is tighter than the real (gloved) hand, so we dilate generously to
    make sure the whole hand region falls inside the inpaint hole."""
    m = _rasterize(verts_cam, faces, K, x1, y1, cw, ch, out_size, shade_uv=None)
    from scipy.ndimage import binary_fill_holes, binary_dilation
    b = binary_fill_holes(m > 127)
    if dilate > 0:
        R = int(dilate)
        yy, xx = np.ogrid[-R:R + 1, -R:R + 1]
        disk = (xx * xx + yy * yy) <= R * R          # isotropic growth
        b = binary_dilation(b, structure=disk)
        b = binary_fill_holes(b)                      # close gaps between fingers
    return (b * 255).astype(np.uint8)


# --------------------------------------------------------------------------
# MANO
# --------------------------------------------------------------------------
def build_mano(mano_dir, is_rhand, device):
    from smplx import MANO
    with monkey_patched_for_chumpy():
        layer = MANO(mano_dir, use_pca=False, flat_hand_mean=False,
                     is_rhand=is_rhand, create_transl=False)
    return layer.to(device)


def mano_forward(layer, rot, pose, trans, shape, device):
    """rot (F,3), pose (F,45), trans (F,3), shape (10,) or (F,10)
    -> verts (F,778,3), joints (F,21,3) world, joints in OpenPose/training order."""
    rot = np.asarray(rot, dtype=np.float32)
    F = rot.shape[0]
    shape = np.asarray(shape, dtype=np.float32)
    if shape.ndim == 1:                      # ARCTIC stores one shape per sequence
        shape = np.tile(shape[None], (F, 1))
    out = layer(
        global_orient=torch.as_tensor(rot, dtype=torch.float32, device=device),
        hand_pose=torch.as_tensor(pose, dtype=torch.float32, device=device),
        betas=torch.as_tensor(shape, dtype=torch.float32, device=device),
        transl=torch.as_tensor(trans, dtype=torch.float32, device=device),
    )
    verts = out.vertices
    joints = out.joints
    if joints.shape[1] == 16:
        # smplx MANO here returns 16 base joints; append the 5 fingertips from the
        # mesh exactly like HaMeR's mano_wrapper (smplx.vertex_ids['mano'] order:
        # thumb, index, middle, ring, pinky -> joints 16..20).
        from smplx.vertex_ids import vertex_ids as _VID
        tip_idx = list(_VID["mano"].values())
        tips = verts[:, tip_idx, :]
        joints = torch.cat([joints, tips], dim=1)
    assert joints.shape[1] == 21, f"expected 21 joints, got {joints.shape[1]}"
    joints = joints[:, MANO_TO_OPENPOSE, :]   # MANO-native -> OpenPose/training order
    return verts.detach().cpu().numpy(), joints.detach().cpu().numpy()


def transform_world2cam(pts, world2cam):
    """pts (N,3) world -> (N,3) cam. world2cam (4,4)."""
    homo = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)], axis=1)
    return (world2cam @ homo.T).T[:, :3]


def distort_pts3d(pts_cam, dist):
    """Numpy port of ARCTIC common.transforms.distort_pts3d_all for the egocentric
    fisheye camera: undistorted cam coords (N,3) -> distorted cam coords (N,3),
    so a linear K-projection lands on the real (distorted) ego pixels. dist: (8,)."""
    pts = pts_cam.astype(np.float64)
    z = pts[:, 2]
    zi = 1.0 / z
    x1, y1 = pts[:, 0] * zi, pts[:, 1] * zi
    x1_2, y1_2, x1y1 = x1 * x1, y1 * y1, x1 * y1
    r2 = x1_2 + y1_2
    r4 = r2 * r2
    r6 = r4 * r2
    r_dist = (1 + dist[0]*r2 + dist[1]*r4 + dist[4]*r6) / \
             (1 + dist[5]*r2 + dist[6]*r4 + dist[7]*r6)
    x2 = x1 * r_dist + 2*dist[2]*x1y1 + dist[3]*(r2 + 2*x1_2)
    y2 = y1 * r_dist + 2*dist[3]*x1y1 + dist[2]*(r2 + 2*y1_2)
    return np.stack([x2 * z, y2 * z, z], axis=1).astype(np.float32)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arctic-root", required=True,
                    help="ARCTIC data root containing raw_seqs/, meta/, images/")
    ap.add_argument("--mano-dir", required=True,
                    help="Dir with MANO_LEFT.pkl / MANO_RIGHT.pkl")
    ap.add_argument("--uv-left", required=True)
    ap.add_argument("--uv-right", required=True)
    ap.add_argument("--subject", default="s01")
    ap.add_argument("--seq", default="box_grab_01")
    ap.add_argument("--view", type=int, default=1, help="Allocentric view 1..8")
    ap.add_argument("--scale", type=float, default=1.5)
    ap.add_argument("--out-size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--sides", default="right,left")
    ap.add_argument("--mask-dilate", type=int, default=24,
                    help="Disk radius in px (at out-size) to grow the hand mask")
    ap.add_argument("--min-inside", type=float, default=0.6,
                    help="Skip a hand if fewer than this fraction of its 2D joints are in-frame")
    ap.add_argument("--out", required=True)
    ap.add_argument("--inspect", action="store_true",
                    help="Print loaded structures and exit (no rendering)")
    args = ap.parse_args()

    root = Path(args.arctic_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- load camera meta ---
    misc = json.loads((root / "meta" / "misc.json").read_text())
    sub_misc = misc[args.subject]
    assert 0 <= args.view <= 8, "view 0 (egocentric) or 1..8 (allocentric)"
    is_ego = (args.view == 0)
    img_w, img_h = sub_misc["image_size"][args.view]   # [W,H] per view (incl ego at 0)
    img_hw = (img_h, img_w)
    ioi_offset = int(sub_misc["ioi_offset"])

    # --- load MANO params ---
    mano_npy = root / "raw_seqs" / args.subject / f"{args.seq}.mano.npy"
    params = np.load(mano_npy, allow_pickle=True).item()

    # --- camera: allocentric (constant) or egocentric (per-frame + distortion) ---
    if is_ego:
        egocam = np.load(root / "raw_seqs" / args.subject / f"{args.seq}.egocam.dist.npy",
                         allow_pickle=True).item()
        R_ego = np.asarray(egocam["R_k_cam_np"], dtype=np.float32)        # (F,3,3)
        T_ego = np.asarray(egocam["T_k_cam_np"], dtype=np.float32).reshape(-1, 3)
        world2ego = np.zeros((R_ego.shape[0], 4, 4), dtype=np.float32)
        world2ego[:, :3, :3] = R_ego
        world2ego[:, :3, 3] = T_ego
        world2ego[:, 3, 3] = 1.0
        K = np.asarray(egocam["intrinsics"], dtype=np.float32)            # (3,3)
        dist8 = np.asarray(egocam["dist8"], dtype=np.float64)             # (8,)
        world2cam = None
    else:
        world2cam = np.array(sub_misc["world2cam"][args.view - 1], dtype=np.float32)  # (4,4)
        K = np.array(sub_misc["intris_mat"][args.view - 1], dtype=np.float32)          # (3,3)
        dist8 = None

    if args.inspect:
        print("misc subject keys:", list(sub_misc.keys()))
        print("world2cam[view-1]:\n", world2cam)
        print("K[view-1]:\n", K)
        print("image_size[view]:", sub_misc["image_size"][args.view], "ioi_offset:", ioi_offset)
        print("mano.npy top keys:", list(params.keys()))
        for hk in params:
            if isinstance(params[hk], dict):
                print(f"  [{hk}] subkeys:", {k: np.asarray(v).shape for k, v in params[hk].items()})
        return

    img_dir = root / "images" / args.subject / args.seq / str(args.view)

    sides = [s for s in args.sides.split(",") if s]
    n_frames_total = len(params[sides[0]]["rot"])

    # MANO layers + UV templates
    layers = {s: build_mano(args.mano_dir, is_rhand=(s == "right"), device=device) for s in sides}
    uv_obj = {"left": args.uv_left, "right": args.uv_right}
    uvft = {s: parse_mano_uv_obj(Path(uv_obj[s])) for s in sides}

    # Precompute world verts/joints for all frames per side
    cache = {}
    for s in sides:
        p = params[s]
        verts_w, joints_w = mano_forward(
            layers[s], p["rot"], p["pose"], p["trans"], p["shape"], device)
        cache[s] = (verts_w, joints_w)
    faces = layers[sides[0]].faces.astype(np.int64)  # (1538,3), shared topology

    frame_ids = list(range(args.start, n_frames_total, args.stride))[: args.max_frames]
    print(f"{args.subject}/{args.seq} view {args.view}: {len(frame_ids)} frames x {len(sides)} hands")

    # Accumulators (data.h5-compatible keys)
    acc = {k: [] for k in ["crops", "masks", "skeletons", "uvs",
                           "kp2d_out", "kp3d", "is_right", "frame_keys", "side", "view"]}

    for g in frame_ids:
        imgnum = g + ioi_offset
        img_path = img_dir / f"{imgnum:05d}.jpg"
        if not img_path.exists():
            print(f"  [skip] missing image {img_path}")
            continue
        frame_rgb = np.array(Image.open(img_path).convert("RGB"))

        for s in sides:
            verts_w, joints_w = cache[s]
            w2c = world2ego[g] if is_ego else world2cam
            verts_cam = transform_world2cam(verts_w[g], w2c)
            joints_cam = transform_world2cam(joints_w[g], w2c)
            if is_ego:                       # apply fisheye distortion to hit real ego pixels
                verts_cam = distort_pts3d(verts_cam, dist8)
                joints_cam = distort_pts3d(joints_cam, dist8)
            kp2d_full = project_to_image(joints_cam, K)

            inside = ((kp2d_full[:, 0] >= 0) & (kp2d_full[:, 0] < img_w) &
                      (kp2d_full[:, 1] >= 0) & (kp2d_full[:, 1] < img_h)).mean()
            if inside < args.min_inside:
                print(f"  [skip] {imgnum:05d} {s}: hand {inside:.0%} in-frame")
                continue

            x1, y1, x2, y2 = keypoint_crop_box(kp2d_full, args.scale, img_hw)
            cw, ch = x2 - x1, y2 - y1
            if cw < 4 or ch < 4:
                continue
            sx, sy = args.out_size / cw, args.out_size / ch
            kp2d_out = (kp2d_full - np.array([x1, y1])) * np.array([sx, sy])

            crop = render_crop(frame_rgb, y1, y2, x1, x2, args.out_size)
            skel = render_skeleton(kp2d_out, args.out_size)
            vt, ft = uvft[s]
            uv = render_uv(verts_cam, faces, vt, ft, K, x1, y1, cw, ch, args.out_size)
            mask = render_mask(verts_cam, faces, K, x1, y1, cw, ch, args.out_size,
                               dilate=args.mask_dilate)

            acc["crops"].append(crop)
            acc["masks"].append(mask)
            acc["skeletons"].append(skel)
            acc["uvs"].append(uv)
            acc["kp2d_out"].append(kp2d_out.astype(np.float32))
            acc["kp3d"].append(joints_cam.astype(np.float32))
            acc["is_right"].append(1 if s == "right" else 0)
            acc["frame_keys"].append(f"{args.subject}/{args.seq}/{args.view}/frame_{imgnum:05d}_{s}")
            acc["side"].append(s)
            acc["view"].append(str(args.view))
        print(f"  frame {imgnum:05d} done")

    n = len(acc["crops"])
    if n == 0:
        raise SystemExit("No samples produced — check image paths / ioi_offset / view.")

    import h5py
    dt_str = h5py.special_dtype(vlen=str)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("crops", data=np.stack(acc["crops"]), dtype=np.uint8, compression="lzf")
        f.create_dataset("masks", data=np.stack(acc["masks"]), dtype=np.uint8, compression="lzf")
        f.create_dataset("skeletons", data=np.stack(acc["skeletons"]), dtype=np.uint8, compression="lzf")
        f.create_dataset("uvs", data=np.stack(acc["uvs"]), dtype=np.uint8, compression="lzf")
        f.create_dataset("keypoints_2d_output", data=np.stack(acc["kp2d_out"]), dtype=np.float32)
        f.create_dataset("keypoints_3d", data=np.stack(acc["kp3d"]), dtype=np.float32)
        f.create_dataset("is_right", data=np.array(acc["is_right"], dtype=np.uint8))
        f.create_dataset("frame_keys", data=np.array(acc["frame_keys"], dtype=object), dtype=dt_str)
        f.create_dataset("side", data=np.array(acc["side"], dtype=object), dtype=dt_str)
        f.create_dataset("view", data=np.array(acc["view"], dtype=object), dtype=dt_str)
        f.attrs["keypoint_names"] = json.dumps(KEYPOINT_NAMES)
        f.attrs["output_size"] = args.out_size
        f.attrs["n_samples"] = n
        f.attrs["source"] = f"ARCTIC {args.subject}/{args.seq} view{args.view}"
    print(f"Wrote {args.out}  ({n} samples)")


if __name__ == "__main__":
    main()
