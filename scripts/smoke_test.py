#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hohs_mano_regressor.config import load_yaml, resolve_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that the HaMeR/ARTIC fine-tune environment is wired.")
    parser.add_argument("--config", default="configs/train_artic.yaml")
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--soft", action="store_true", help="Print failures without returning a non-zero exit code.")
    args = parser.parse_args()

    failures: list[str] = []
    cfg = load_yaml(resolve_path(args.config, ROOT))
    hamer_root = resolve_path(cfg["paths"]["hamer_root"], ROOT)
    checkpoint = resolve_path(cfg["paths"]["hamer_checkpoint"], ROOT)
    train_npz = resolve_path(cfg["paths"]["train_npz"], ROOT)
    image_root = resolve_path(cfg["paths"]["image_root"], ROOT)
    mano_path = hamer_root / "_DATA/data/mano/MANO_RIGHT.pkl"

    print(f"Repo root: {ROOT}")
    print(f"HaMeR root: {hamer_root}")
    print(f"Checkpoint: {checkpoint}")
    print(f"ARTIC train NPZ: {train_npz}")
    print(f"Image root: {image_root}")

    try:
        import torch

        print(f"Torch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception as exc:  # pragma: no cover - diagnostic script
        failures.append(f"torch import failed: {exc}")

    if not hamer_root.exists():
        failures.append(f"missing HaMeR checkout: {hamer_root}")
    else:
        sys.path.insert(0, str(hamer_root))
        try:
            import hamer  # noqa: F401

            print("HaMeR import: ok")
        except Exception as exc:  # pragma: no cover - diagnostic script
            failures.append(f"hamer import failed: {exc}")

    for path, label in (
        (checkpoint, "HaMeR checkpoint"),
        (mano_path, "MANO_RIGHT.pkl"),
        (train_npz, "ARTIC train NPZ"),
        (image_root, "ARTIC image root"),
    ):
        if not path.exists():
            failures.append(f"missing {label}: {path}")

    if args.load_model and not failures:
        from hohs_mano_regressor.train_artic import main as train_main

        train_main(["--config", str(resolve_path(args.config, ROOT)), "--fast-dev-run"])

    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  - {failure}")
        return 0 if args.soft else 1
    print("Smoke check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

