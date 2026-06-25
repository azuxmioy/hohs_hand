from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ALIASES: dict[str, tuple[str, ...]] = {
    "imgname": ("imgname", "image_path", "image_paths", "image", "image_file", "rgb_path"),
    "keypoints_2d": ("hand_keypoints_2d", "keypoints_2d", "joints2d", "joints_2d", "uv"),
    "keypoints_3d": ("hand_keypoints_3d", "keypoints_3d", "joints3d", "joints_3d", "xyz"),
    "bbox_xyxy": ("bbox_xyxy", "bbox", "hand_bbox", "box_xyxy"),
    "bbox_xywh": ("bbox_xywh", "box_xywh"),
    "center": ("center", "bbox_center", "box_center"),
    "scale": ("scale", "bbox_size", "box_size"),
    "global_orient": ("global_orient", "root_orient", "mano_global_orient"),
    "hand_pose": ("hand_pose", "mano_hand_pose", "mano_pose", "pose"),
    "betas": ("betas", "shape", "mano_betas"),
    "right": ("right", "is_right", "hand_is_right", "hand_side", "side"),
    "split": ("split", "subset"),
}


@dataclass(frozen=True)
class ConversionSummary:
    samples: int
    output: Path
    missing_mano_pose: int
    missing_betas: int


def convert_artic_to_hamer_npz(
    source: str | Path,
    output: str | Path,
    image_root: str | Path | None = None,
    split: str | None = None,
    bbox_padding: float = 1.0,
    limit: int | None = None,
    allow_missing_3d: bool = False,
) -> ConversionSummary:
    records = list(_iter_records(source))
    if split is not None:
        records = [record for record in records if str(_lookup(record, "split", "")) == split]
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError(f"No samples found in {source}")

    img_names: list[str] = []
    centers: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    keypoints_2d: list[np.ndarray] = []
    keypoints_3d: list[np.ndarray] = []
    hand_poses: list[np.ndarray] = []
    has_hand_pose: list[float] = []
    betas: list[np.ndarray] = []
    has_betas: list[float] = []
    right: list[float] = []
    extra_info: list[dict[str, Any]] = []
    missing_pose_count = 0
    missing_betas_count = 0

    image_root_path = Path(image_root).expanduser().resolve() if image_root else None

    for source_index, record in enumerate(records):
        img_names.append(_normalise_imgname(_lookup_required(record, "imgname"), image_root_path))

        kp2d = _points_with_conf(_lookup(record, "keypoints_2d", None), dims=2, name="keypoints_2d")
        kp3d_raw = _lookup(record, "keypoints_3d", None)
        if kp3d_raw is None and not allow_missing_3d:
            raise ValueError(f"Sample {source_index} is missing 3D hand keypoints")
        kp3d = _points_with_conf(kp3d_raw, dims=3, name="keypoints_3d")

        center, scale = _center_scale(record, kp2d, bbox_padding)
        pose, has_pose = _mano_pose(record)
        shape, has_shape = _betas(record)

        if not has_pose:
            missing_pose_count += 1
        if not has_shape:
            missing_betas_count += 1

        keypoints_2d.append(kp2d)
        keypoints_3d.append(kp3d)
        centers.append(center)
        scales.append(scale)
        hand_poses.append(pose)
        has_hand_pose.append(float(has_pose))
        betas.append(shape)
        has_betas.append(float(has_shape))
        right.append(_right_flag(_lookup(record, "right", 1.0)))
        extra_info.append({"source_index": int(source_index)})

    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        imgname=np.asarray(img_names, dtype=object),
        center=np.asarray(centers, dtype=np.float32),
        scale=np.asarray(scales, dtype=np.float32),
        hand_keypoints_2d=np.asarray(keypoints_2d, dtype=np.float32),
        hand_keypoints_3d=np.asarray(keypoints_3d, dtype=np.float32),
        hand_pose=np.asarray(hand_poses, dtype=np.float32),
        has_hand_pose=np.asarray(has_hand_pose, dtype=np.float32),
        betas=np.asarray(betas, dtype=np.float32),
        has_betas=np.asarray(has_betas, dtype=np.float32),
        right=np.asarray(right, dtype=np.float32),
        extra_info=np.asarray(extra_info, dtype=object),
    )
    return ConversionSummary(
        samples=len(img_names),
        output=output_path,
        missing_mano_pose=missing_pose_count,
        missing_betas=missing_betas_count,
    )


def _iter_records(source: str | Path) -> Iterable[Mapping[str, Any]]:
    path = Path(source).expanduser()
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as data:
            if "records" in data:
                records = data["records"]
                for record in records:
                    yield _as_record(record)
                return
            arrays = {key: data[key] for key in data.files}
        length = _infer_length(arrays)
        for index in range(length):
            yield {key: _value_at(value, index) for key, value in arrays.items()}
        return

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload = payload.get("samples", payload.get("records", payload))
            if isinstance(payload, dict):
                arrays = payload
                length = _infer_length(arrays)
                for index in range(length):
                    yield {key: _value_at(value, index) for key, value in arrays.items()}
                return
        for record in payload:
            yield _as_record(record)
        return

    raise ValueError(f"Unsupported annotation format: {path.suffix}")


def _infer_length(arrays: Mapping[str, Any]) -> int:
    for canonical, aliases in ALIASES.items():
        del canonical
        for key in aliases:
            if key in arrays:
                return len(arrays[key])
    raise ValueError("Could not infer number of samples from annotation arrays")


def _as_record(record: Any) -> Mapping[str, Any]:
    if isinstance(record, np.ndarray) and record.shape == ():
        record = record.item()
    if isinstance(record, Mapping):
        return record
    raise TypeError(f"Expected record mapping, got {type(record)!r}")


def _value_at(value: Any, index: int) -> Any:
    array = np.asarray(value, dtype=object if isinstance(value, list) else None)
    return array[index]


def _lookup(record: Mapping[str, Any], canonical: str, default: Any = None) -> Any:
    for key in ALIASES[canonical]:
        if key in record:
            value = record[key]
            if isinstance(value, np.ndarray) and value.shape == ():
                return value.item()
            return value
    return default


def _lookup_required(record: Mapping[str, Any], canonical: str) -> Any:
    value = _lookup(record, canonical, None)
    if value is None:
        raise ValueError(f"Missing required field for {canonical}: {ALIASES[canonical]}")
    return value


def _normalise_imgname(value: Any, image_root: Path | None) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray):
        value = value.item()
    path = Path(str(value))
    if image_root is not None:
        try:
            return str(path.expanduser().resolve().relative_to(image_root))
        except ValueError:
            pass
    return str(path)


def _points_with_conf(value: Any, dims: int, name: str) -> np.ndarray:
    if value is None:
        return np.zeros((21, dims + 1), dtype=np.float32)
    points = np.asarray(value, dtype=np.float32)
    if points.ndim == 1:
        if points.size == 21 * dims:
            points = points.reshape(21, dims)
        elif points.size == 21 * (dims + 1):
            points = points.reshape(21, dims + 1)
    if points.shape[-1] < dims:
        raise ValueError(f"{name} must have at least {dims} coordinates, got {points.shape}")
    points = points.reshape(-1, points.shape[-1])
    coords = points[:, :dims]
    if points.shape[-1] > dims:
        conf = points[:, dims : dims + 1]
    else:
        conf = np.ones((points.shape[0], 1), dtype=np.float32)
    if coords.shape[0] != 21:
        raise ValueError(f"{name} must contain 21 hand joints, got {coords.shape[0]}")
    return np.concatenate([coords, conf], axis=1).astype(np.float32)


def _center_scale(
    record: Mapping[str, Any],
    keypoints_2d: np.ndarray,
    bbox_padding: float,
) -> tuple[np.ndarray, np.ndarray]:
    center = _lookup(record, "center", None)
    scale = _lookup(record, "scale", None)
    if center is not None and scale is not None:
        return _vector(center, 2, "center"), _scale_vector(scale)

    bbox_xyxy = _lookup(record, "bbox_xyxy", None)
    if bbox_xyxy is not None:
        box = _vector(bbox_xyxy, 4, "bbox_xyxy")
        x1, y1, x2, y2 = box
        width = max(float(x2 - x1), 1.0)
        height = max(float(y2 - y1), 1.0)
        return (
            np.asarray([x1 + width * 0.5, y1 + height * 0.5], dtype=np.float32),
            np.asarray([width * bbox_padding, height * bbox_padding], dtype=np.float32),
        )

    bbox_xywh = _lookup(record, "bbox_xywh", None)
    if bbox_xywh is not None:
        box = _vector(bbox_xywh, 4, "bbox_xywh")
        x, y, width, height = box
        width = max(float(width), 1.0)
        height = max(float(height), 1.0)
        return (
            np.asarray([x + width * 0.5, y + height * 0.5], dtype=np.float32),
            np.asarray([width * bbox_padding, height * bbox_padding], dtype=np.float32),
        )

    valid = keypoints_2d[:, 2] > 0
    if valid.sum() < 2:
        raise ValueError("Need center/scale, bbox, or at least two visible 2D joints")
    xy = keypoints_2d[valid, :2]
    low = xy.min(axis=0)
    high = xy.max(axis=0)
    size = np.maximum(high - low, 1.0) * bbox_padding
    return ((low + high) * 0.5).astype(np.float32), size.astype(np.float32)


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size != size:
        raise ValueError(f"{name} must have {size} values, got {vector.size}")
    return vector


def _scale_vector(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size == 1:
        return np.repeat(vector, 2).astype(np.float32)
    if vector.size == 2:
        return vector.astype(np.float32)
    raise ValueError(f"scale must have one or two values, got {vector.size}")


def _mano_pose(record: Mapping[str, Any]) -> tuple[np.ndarray, bool]:
    pose = _lookup(record, "hand_pose", None)
    global_orient = _lookup(record, "global_orient", None)
    if pose is None and global_orient is None:
        return np.zeros(48, dtype=np.float32), False
    if pose is None:
        pose_values = np.zeros(45, dtype=np.float32)
    else:
        pose_values = np.asarray(pose, dtype=np.float32).reshape(-1)
    if pose_values.size == 48:
        return pose_values.astype(np.float32), True
    if pose_values.size != 45:
        raise ValueError(f"hand_pose must have 45 or 48 axis-angle values, got {pose_values.size}")
    orient = np.zeros(3, dtype=np.float32) if global_orient is None else _vector(global_orient, 3, "global_orient")
    return np.concatenate([orient, pose_values]).astype(np.float32), True


def _betas(record: Mapping[str, Any]) -> tuple[np.ndarray, bool]:
    value = _lookup(record, "betas", None)
    if value is None:
        return np.zeros(10, dtype=np.float32), False
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size < 10:
        vector = np.pad(vector, (0, 10 - vector.size))
    return vector[:10].astype(np.float32), True


def _right_flag(value: Any) -> float:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return 1.0 if value.lower() in {"right", "r", "1", "true"} else 0.0
    return float(bool(value))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert ARTIC hand annotations to HaMeR ImageDataset NPZ.")
    parser.add_argument("--source", required=True, help="Input NPZ, JSON, or JSONL ARTIC annotation manifest.")
    parser.add_argument("--output", required=True, help="Output HaMeR-compatible NPZ path.")
    parser.add_argument("--image-root", default=None, help="Image root used to make image paths relative.")
    parser.add_argument("--split", default=None, help="Optional split value to keep, if the manifest contains split labels.")
    parser.add_argument("--bbox-padding", type=float, default=1.0, help="Padding applied when deriving boxes.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample cap for quick checks.")
    parser.add_argument("--allow-missing-3d", action="store_true", help="Fill missing 3D joints with zeros.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    summary = convert_artic_to_hamer_npz(
        source=args.source,
        output=args.output,
        image_root=args.image_root,
        split=args.split,
        bbox_padding=args.bbox_padding,
        limit=args.limit,
        allow_missing_3d=args.allow_missing_3d,
    )
    print(
        "Wrote "
        f"{summary.samples} samples to {summary.output} "
        f"(missing MANO pose: {summary.missing_mano_pose}, missing betas: {summary.missing_betas})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
