"""Generate cropped hand images from MANO results and tight masks.

Output in hand_crops/<segment>/<view>/ (flat folder, all images output_size×output_size):
  frame_XXXXXX_<side>_crop.png      - cropped original RGB
  frame_XXXXXX_<side>_mask.png      - cropped tight mask (binary, grayscale)
  frame_XXXXXX_<side>_skeleton.png  - MANO 2D skeleton on black
  frame_XXXXXX_<side>_uv.png        - MANO mesh UV-colored (u→R, v→G) on black
  bbox.json   - all frames: {"frame_XXXXXX_side": {x1,y1,x2,y2,...}}
  info.json   - all frames: {"frame_XXXXXX_side": {side, kp2d, kp3d,...}}

With --h5:
  Single segment → hand_crops/<segment>/<view>/data.h5
  With --all-segments → hand_crops/data.h5  (one combined file for all segments/views)
  frame_keys include segment/view: "<segment>/<view>/frame_XXXXXX_<side>"
"""

import argparse
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent

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
    "thumb_mcp",  "thumb_pip",  "thumb_dip",  "thumb_tip",
    "index_mcp",  "index_pip",  "index_dip",  "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp",   "ring_pip",   "ring_dip",   "ring_tip",
    "pinky_mcp",  "pinky_pip",  "pinky_dip",  "pinky_tip",
]


# ---------------------------------------------------------------------------
# OBJ UV parsing
# ---------------------------------------------------------------------------

def parse_mano_uv_obj(obj_path: Path):
    """Parse MANO UV OBJ → vt (V_uv,2) float32, ft (F,3) int32 UV indices (0-based)."""
    vt, ft = [], []
    with open(obj_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("vt "):
                parts = line.split()
                vt.append((float(parts[1]), float(parts[2])))
            elif line.startswith("f "):
                triplets = line.split()[1:]
                ft.append([int(t.split("/")[1]) - 1 for t in triplets])
    return np.array(vt, dtype=np.float32), np.array(ft, dtype=np.int32)


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def get_video_fps(video_path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "json", str(video_path)],
        capture_output=True, text=True,
    )
    frac = json.loads(r.stdout)["streams"][0]["r_frame_rate"]
    num, den = frac.split("/")
    return float(num) / float(den)


def extract_frames(video_path: Path, step: int, max_frames: int = None) -> dict:
    """Extract every `step`-th frame via ffmpeg pipe. Returns {frame_idx: np.array}."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames", "-of", "json", str(video_path)],
        capture_output=True, text=True,
    )
    info = json.loads(r.stdout)["streams"][0]
    w, h, total = info["width"], info["height"], int(info["nb_frames"])

    cmd = ["ffmpeg", "-i", str(video_path),
           "-vf", f"select=not(mod(n\\,{step}))", "-vsync", "0"]
    if max_frames is not None:
        cmd += ["-vframes", str(max_frames)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    result = subprocess.run(cmd, capture_output=True)
    raw = np.frombuffer(result.stdout, dtype=np.uint8)
    n = len(raw) // (h * w * 3)
    frames = raw[: n * h * w * 3].reshape(n, h, w, 3)
    indices = list(range(0, total, step))[:n]
    return {fi: frames[i] for i, fi in enumerate(indices)}


# ---------------------------------------------------------------------------
# Stereo calibration
# ---------------------------------------------------------------------------

def load_stereo_calib(segment: str):
    """Load T_right_bleft (4×4) and K_right (3×3) for a segment.

    Returns R (3×3), t (3,), K_right (3×3) — all float32.
    These transform 3-D points from bleft camera space to right camera space:
        p_right = R @ p_bleft + t
    """
    npz_path = (ROOT / "results" / "stereo_triangulation_filtered3d"
                / segment / "best_triangulated_joints.npz")
    T = np.load(npz_path)["T_right_bleft"].astype(np.float32)
    R, t = T[:3, :3], T[:3, 3]

    K_right_path = (ROOT / "calibration" / "pinhole_intrinsics"
                    / segment / "right_pinhole_K.json")
    K_right = np.array(json.loads(K_right_path.read_text())["K"], dtype=np.float32)
    return R, t, K_right


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def keypoint_crop_box(kp2d: np.ndarray, scale: float, img_hw: tuple) -> tuple:
    """Square crop box centered on 2D keypoint extent, expanded by scale."""
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


def project_to_image(verts_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project (N,3) points already in camera space → (N,2) image coords."""
    x = K[0, 0] * verts_cam[:, 0] / verts_cam[:, 2] + K[0, 2]
    y = K[1, 1] * verts_cam[:, 1] / verts_cam[:, 2] + K[1, 2]
    return np.stack([x, y], axis=1)


# ---------------------------------------------------------------------------
# Renderers  (all return np.ndarray uint8)
# ---------------------------------------------------------------------------

def render_crop(frame_rgb: np.ndarray, y1: int, y2: int, x1: int, x2: int,
                out_size: int) -> np.ndarray:
    arr = frame_rgb[y1:y2, x1:x2]
    return np.array(Image.fromarray(arr).resize((out_size, out_size), Image.LANCZOS))


def fill_mask_holes(binary: np.ndarray) -> np.ndarray:
    """Fill interior holes (e.g. marker artifacts) via binary_fill_holes."""
    from scipy.ndimage import binary_fill_holes
    return binary_fill_holes(binary).astype(binary.dtype)


def render_mask(mask_path: Path, y1: int, y2: int, x1: int, x2: int,
                out_size: int) -> np.ndarray:
    """Load mask PNG, fill holes, crop, resize → (out_size, out_size) uint8 (0 or 255)."""
    if mask_path.exists():
        mask = np.array(Image.open(mask_path).convert("L"))
        binary = (mask > 127).astype(np.uint8)
        binary = fill_mask_holes(binary)
        crop = (binary * 255)[y1:y2, x1:x2]
    else:
        crop = np.full((y2 - y1, x2 - x1), 255, dtype=np.uint8)
    resized = np.array(Image.fromarray(crop).resize((out_size, out_size), Image.NEAREST))
    return (resized > 127).astype(np.uint8) * 255


def render_skeleton(kp2d: np.ndarray, out_size: int) -> np.ndarray:
    """Skeleton on black → (out_size, out_size, 3) uint8."""
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


def render_uv(
    verts_cam: np.ndarray, faces: np.ndarray,
    vt: np.ndarray, ft: np.ndarray,
    K: np.ndarray,
    x1: int, y1: int, cw: int, ch: int,
    out_size: int,
) -> np.ndarray:
    """UV-colored mesh (u→R, v→G, B=0) on black → (out_size, out_size, 3) uint8.
    verts_cam: (V, 3) vertices already in camera space."""
    verts_2d_full = project_to_image(verts_cam, K)
    verts_2d = (verts_2d_full - np.array([x1, y1], dtype=np.float32)) \
               * np.array([out_size / cw, out_size / ch], dtype=np.float32)

    order = np.argsort(verts_cam[:, 2][faces].mean(axis=1))[::-1]  # back to front

    img = np.zeros((out_size, out_size, 3), dtype=np.uint8)

    for fi in order:
        f = faces[fi]
        p = verts_2d[f]       # (3, 2)
        uv = vt[ft[fi]]       # (3, 2)

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

        u = (w0[inside]*uv[0,0] + w1[inside]*uv[1,0] + w2[inside]*uv[2,0]).clip(0, 1)
        v = (w0[inside]*uv[0,1] + w1[inside]*uv[1,1] + w2[inside]*uv[2,1]).clip(0, 1)
        ix = px[inside].astype(np.int32)
        iy = py[inside].astype(np.int32)
        img[iy, ix, 0] = (u * 255).astype(np.uint8)
        img[iy, ix, 1] = (v * 255).astype(np.uint8)

    return img


# ---------------------------------------------------------------------------
# H5 writer
# ---------------------------------------------------------------------------

class H5Writer:
    """Accumulates per-frame arrays and writes a compact HDF5 dataset file."""

    def __init__(self, path: Path, out_size: int):
        self.path = path
        self.out_size = out_size
        self._keys:     list[str] = []
        self._segments: list[str] = []
        self._views:    list[str] = []
        self._crops:     list[np.ndarray] = []
        self._masks:     list[np.ndarray] = []
        self._skeletons: list[np.ndarray] = []
        self._uvs:       list[np.ndarray] = []
        self._kp2d_out:  list[np.ndarray] = []
        self._kp2d_full: list[np.ndarray] = []
        self._kp3d:      list[np.ndarray] = []
        self._sides:     list[str] = []
        self._is_right:  list[int] = []
        self._frame_idx: list[int] = []
        self._bboxes:    list[list] = []

    def add(self, key: str, segment: str, view: str,
            crop: np.ndarray, mask: np.ndarray,
            skeleton: np.ndarray, uv: np.ndarray,
            kp2d_out: np.ndarray, kp2d_full: np.ndarray, kp3d: np.ndarray,
            side: str, is_right: int, frame_idx: int, bbox: tuple):
        self._keys.append(key)
        self._segments.append(segment)
        self._views.append(view)
        self._crops.append(crop)
        self._masks.append(mask)
        self._skeletons.append(skeleton)
        self._uvs.append(uv)
        self._kp2d_out.append(kp2d_out)
        self._kp2d_full.append(kp2d_full)
        self._kp3d.append(kp3d)
        self._sides.append(side)
        self._is_right.append(is_right)
        self._frame_idx.append(frame_idx)
        self._bboxes.append(list(bbox))

    def write(self):
        N = len(self._keys)
        S = self.out_size
        dt_str = h5py.special_dtype(vlen=str)
        # Use "a" (append) so multiple segments can be written incrementally
        # but here we batch-write everything at once.
        with h5py.File(self.path, "w") as f:
            f.create_dataset("frame_keys", data=np.array(self._keys,     dtype=object), dtype=dt_str)
            f.create_dataset("segment",    data=np.array(self._segments, dtype=object), dtype=dt_str)
            f.create_dataset("view",       data=np.array(self._views,    dtype=object), dtype=dt_str)
            f.create_dataset("crops",      data=np.stack(self._crops),     dtype=np.uint8, compression="lzf")
            f.create_dataset("masks",      data=np.stack(self._masks),     dtype=np.uint8, compression="lzf")
            f.create_dataset("skeletons",  data=np.stack(self._skeletons), dtype=np.uint8, compression="lzf")
            f.create_dataset("uvs",        data=np.stack(self._uvs),       dtype=np.uint8, compression="lzf")
            f.create_dataset("keypoints_2d_output",     data=np.stack(self._kp2d_out),  dtype=np.float32)
            f.create_dataset("keypoints_2d_full_image", data=np.stack(self._kp2d_full), dtype=np.float32)
            f.create_dataset("keypoints_3d",            data=np.stack(self._kp3d),      dtype=np.float32)
            f.create_dataset("side",      data=np.array(self._sides,    dtype=object), dtype=dt_str)
            f.create_dataset("is_right",  data=np.array(self._is_right, dtype=np.uint8))
            f.create_dataset("frame_idx", data=np.array(self._frame_idx, dtype=np.int32))
            f.create_dataset("bbox",      data=np.array(self._bboxes,   dtype=np.int32))
            f.attrs["keypoint_names"] = json.dumps(KEYPOINT_NAMES)
            f.attrs["output_size"] = S
            f.attrs["n_samples"] = N
        print(f"  H5 written: {self.path}  ({N} samples)")


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_segment(
    segment: str,
    view: str,
    target_fps: float,
    scale: float,
    out_size: int,
    out_root: Path,
    h5w: "H5Writer | None" = None,
    n_frames: int = None,
):
    """Process one segment/view. If h5w is provided, accumulate into it (no local H5)."""
    mano_dir   = ROOT / "results" / "stereo_mano_bleft" / segment
    mask_base  = ROOT / "tight_masks" / segment / view / "masks"
    video_path = ROOT / "videos" / "pinhole" / segment / f"{view}_pinhole_fov100.mp4"
    out_dir    = out_root / segment / view
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Segment: {segment}  view: {view}")

    uv_obj = {
        "left":  ROOT / "MANO_UV_left.obj",
        "right": ROOT / "MANO_UV_right.obj",
    }

    left_data  = torch.load(mano_dir / "left"  / "mano.pt", map_location="cpu", weights_only=False)
    right_data = torch.load(mano_dir / "right" / "mano.pt", map_location="cpu", weights_only=False)

    # Stereo calibration: only needed for right camera view
    if view == "right":
        R_rl, t_rl, K_proj = load_stereo_calib(segment)
    else:
        R_rl = t_rl = None
        K_proj = np.array(left_data["K_constant"])  # bleft camera K

    all_fi    = np.asarray(left_data["frame_idx"]).tolist()
    video_fps = get_video_fps(video_path)
    step      = max(1, round(video_fps / target_fps))
    frame_indices = all_fi[::step]
    if n_frames is not None:
        frame_indices = frame_indices[:n_frames]
    print(f"Video {video_fps:.0f}fps → target {target_fps}fps → step {step} → {len(frame_indices)} frames")

    print("Extracting frames from video...")
    frames = extract_frames(video_path, step, max_frames=len(frame_indices))

    img_hw = left_data["img_hw"]
    all_bbox, all_info = {}, {}

    for side, mano_data in [("left", left_data), ("right", right_data)]:
        vt, ft = parse_mano_uv_obj(uv_obj[side])
        mask_dir = mask_base / f"{side}_hand"

        fi_all    = np.asarray(mano_data["frame_idx"])
        fi_to_pos = {int(fi): pos for pos, fi in enumerate(fi_all)}
        faces     = np.array(mano_data["faces"])

        # bleft view: use HaMeR 2D keypoints + pred_vertices projected with bleft K
        # right view: use stereo 2D keypoints (right camera) + stereo 3D verts
        #             transformed to right camera space
        if view == "bleft":
            kp2d_all_arr  = np.array(mano_data["pred_keypoints_2d"])       # (N,21,2) bleft image
            verts3d_all   = np.array(mano_data["pred_vertices"])            # (N,778,3) model space
            cam_t_all     = np.array(mano_data["pred_cam_t"])               # (N,3)
            kp3d_all_arr  = np.array(mano_data["pred_keypoints_3d"])        # (N,21,3)
        else:
            kp2d_all_arr  = np.array(mano_data["stereo3d_keypoints2d_right"])  # (N,21,2) right image
            verts3d_all   = np.array(mano_data["stereo3d_vertices_bleft"])      # (N,778,3) bleft cam space
            kp3d_bleft    = np.array(mano_data["stereo3d_keypoints_bleft"])     # (N,21,3) bleft cam space

        print(f"\nProcessing {side} hand ({view} view)...")
        for fi in frame_indices:
            if fi not in fi_to_pos:
                continue
            pos       = fi_to_pos[fi]
            local_key = f"frame_{fi:06d}_{side}"
            full_key  = f"{segment}/{view}/{local_key}"

            kp2d_full = kp2d_all_arr[pos]   # (21,2) in the correct camera's image space

            # Compute camera-space vertices for rendering
            if view == "bleft":
                cam_t     = cam_t_all[pos]
                verts_cam = verts3d_all[pos] + cam_t        # (778,3) bleft cam space
                kp3d      = kp3d_all_arr[pos]
            else:
                v_bleft   = verts3d_all[pos]                # (778,3) bleft cam space
                verts_cam = (R_rl @ v_bleft.T).T + t_rl    # (778,3) right cam space
                # Also transform keypoints to right cam space for kp3d
                kp3d_b    = kp3d_bleft[pos]                 # (21,3)
                kp3d      = (R_rl @ kp3d_b.T).T + t_rl     # (21,3) right cam space

            bx1, by1, bx2, by2 = keypoint_crop_box(kp2d_full, scale, img_hw)
            cw, ch = bx2 - bx1, by2 - by1

            sx, sy = out_size / cw, out_size / ch
            kp2d_out = (kp2d_full - np.array([bx1, by1])) * np.array([sx, sy])

            crop_arr  = render_crop(frames[fi], by1, by2, bx1, bx2, out_size)
            mask_arr  = render_mask(mask_dir / f"mask_{fi:06d}.png",
                                    by1, by2, bx1, bx2, out_size)
            skel_arr  = render_skeleton(kp2d_out, out_size)
            uv_arr    = render_uv(verts_cam, faces, vt, ft, K_proj,
                                  bx1, by1, cw, ch, out_size)

            # Save individual PNGs
            Image.fromarray(crop_arr).save(out_dir / f"{local_key}_crop.png")
            Image.fromarray(mask_arr, mode="L").save(out_dir / f"{local_key}_mask.png")
            Image.fromarray(skel_arr).save(out_dir / f"{local_key}_skeleton.png")
            Image.fromarray(uv_arr).save(out_dir / f"{local_key}_uv.png")

            all_bbox[local_key] = {
                "x1": bx1, "y1": by1, "x2": bx2, "y2": by2,
                "width": cw, "height": ch,
                "full_image_hw": list(img_hw),
                "output_size": out_size, "scale": scale,
            }
            all_info[local_key] = {
                "side": side, "is_right": int(mano_data["is_right"]),
                "frame_idx": fi, "view": view,
                "keypoints_2d_output": kp2d_out.tolist(),
                "keypoints_2d_full_image": kp2d_full.tolist(),
                "keypoints_3d": kp3d.tolist(),
                "keypoint_names": KEYPOINT_NAMES,
            }

            if h5w is not None:
                h5w.add(
                    key=full_key, segment=segment, view=view,
                    crop=crop_arr, mask=mask_arr,
                    skeleton=skel_arr, uv=uv_arr,
                    kp2d_out=kp2d_out.astype(np.float32),
                    kp2d_full=kp2d_full.astype(np.float32),
                    kp3d=kp3d.astype(np.float32),
                    side=side, is_right=int(mano_data["is_right"]),
                    frame_idx=fi, bbox=(bx1, by1, bx2, by2),
                )

            print(f"  {local_key}")

    (out_dir / "bbox.json").write_text(json.dumps(all_bbox, indent=2))
    (out_dir / "info.json").write_text(json.dumps(all_info, indent=2))
    try:
        label = out_dir.relative_to(ROOT)
    except ValueError:
        label = out_dir
    print(f"Saved {len(all_bbox)} entries → {label}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment", default="01_0519_03_all",
                        help="Single segment to process")
    parser.add_argument("--all-segments", action="store_true",
                        help="Process all segments (overrides --segment)")
    parser.add_argument("--view", default="bleft", choices=["bleft", "right"],
                        help="Camera view (ignored with --all-segments, which does both)")
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--out-size", type=int, default=512)
    parser.add_argument("--n-frames", type=int, default=None)
    parser.add_argument("--n-segments", type=int, default=None,
                        help="Limit number of segments when using --all-segments")
    parser.add_argument("--h5", action="store_true",
                        help="Write data.h5. Single segment → per-segment H5; "
                             "--all-segments → one combined data.h5 at out-dir root")
    parser.add_argument("--out-dir", default=str(ROOT / "hand_crops"))
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    kwargs = dict(target_fps=args.fps, scale=args.scale, out_size=args.out_size,
                  n_frames=args.n_frames)

    if args.all_segments:
        # Discover all available segments from results directory
        seg_root = ROOT / "results" / "stereo_mano_bleft"
        segments = sorted(p.name for p in seg_root.iterdir() if p.is_dir())
        if args.n_segments is not None:
            segments = segments[:args.n_segments]
        views    = ["bleft", "right"]
        print(f"Processing {len(segments)} segments × {len(views)} views")

        h5w = H5Writer(out_root / "data.h5", args.out_size) if args.h5 else None
        for seg in segments:
            for view in views:
                try:
                    process_segment(seg, view, out_root=out_root, h5w=h5w, **kwargs)
                except Exception as e:
                    print(f"  [skip] {seg}/{view}: {e}")
        if h5w is not None:
            h5w.write()
    else:
        h5w = None
        if args.h5:
            h5_path = out_root / args.segment / args.view / "data.h5"
            h5_path.parent.mkdir(parents=True, exist_ok=True)
            h5w = H5Writer(h5_path, args.out_size)
        process_segment(args.segment, args.view, out_root=out_root, h5w=h5w, **kwargs)
        if h5w is not None:
            h5w.write()


if __name__ == "__main__":
    main()
