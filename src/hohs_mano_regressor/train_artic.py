from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from .config import load_yaml, project_root, resolve_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune HaMeR on converted ARTIC annotations.")
    parser.add_argument("--config", default="configs/train_artic.yaml", help="Training config YAML.")
    parser.add_argument("--fast-dev-run", action="store_true", help="Run one train/val batch for plumbing checks.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    root = project_root()
    config_path = resolve_path(args.config, root)
    if config_path is None:
        raise ValueError("Missing config path")
    cfg = load_yaml(config_path)

    hamer_root = _required_path(cfg, "paths", "hamer_root", root=root)
    if not hamer_root.exists():
        raise FileNotFoundError(f"HaMeR checkout not found at {hamer_root}. Run scripts/bootstrap_hamer.sh first.")
    sys.path.insert(0, str(hamer_root))

    import hamer.configs as hamer_configs
    from hamer.datasets.image_dataset import ImageDataset

    from .models.artic_hamer import ArticHAMERFineTuner, load_hamer_weights

    hamer_configs.CACHE_DIR_HAMER = str(hamer_root / "_DATA")
    checkpoint_path = _required_path(cfg, "paths", "hamer_checkpoint", root=root)
    model_config_path = resolve_path(cfg["paths"].get("hamer_model_config"), root)
    if model_config_path is None:
        model_config_path = checkpoint_path.parent.parent / "model_config.yaml"
    if not model_config_path.exists():
        raise FileNotFoundError(f"HaMeR model config not found at {model_config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"HaMeR checkpoint not found at {checkpoint_path}")

    model_cfg = hamer_configs.get_config(str(model_config_path), update_cachedir=True)
    _apply_hamer_overrides(model_cfg, cfg)

    seed = cfg.get("seed")
    if seed is not None:
        pl.seed_everything(int(seed), workers=True)

    train_npz = _required_path(cfg, "paths", "train_npz", root=root)
    val_npz = resolve_path(cfg["paths"].get("val_npz"), root)
    image_root = _required_path(cfg, "paths", "image_root", root=root)
    output_dir = _required_path(cfg, "paths", "output_dir", root=root)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "train_artic.yaml")

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(train_cfg.get("num_workers", 4))
    prefetch_factor = int(train_cfg.get("prefetch_factor", 2))
    rescale_factor = float(data_cfg.get("rescale_factor", 2.0))

    train_dataset = ImageDataset(
        model_cfg,
        dataset_file=str(train_npz),
        img_dir=str(image_root),
        train=True,
        rescale_factor=rescale_factor,
    )
    val_dataset = None
    if val_npz is not None and val_npz.exists():
        val_dataset = ImageDataset(
            model_cfg,
            dataset_file=str(val_npz),
            img_dir=str(image_root),
            train=False,
            rescale_factor=rescale_factor,
        )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "drop_last": True,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs) if val_dataset is not None else None

    model = ArticHAMERFineTuner(
        model_cfg,
        freeze_backbone=bool(train_cfg.get("freeze_backbone", True)),
        init_renderer=False,
    )
    load_report = load_hamer_weights(model, checkpoint_path)
    print(
        "Loaded HaMeR checkpoint with "
        f"{len(load_report['missing'])} missing and {len(load_report['unexpected'])} unexpected keys."
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        every_n_train_steps=int(train_cfg.get("checkpoint_every_n_steps", 500)),
        save_last=True,
        save_top_k=1,
    )
    logger = TensorBoardLogger(save_dir=str(output_dir), name="tensorboard", version="")
    trainer = pl.Trainer(
        accelerator=train_cfg.get("accelerator", "auto"),
        devices=train_cfg.get("devices", 1),
        precision=train_cfg.get("precision", "16-mixed"),
        max_steps=int(train_cfg.get("max_steps", 10000)),
        logger=logger,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="step")],
        log_every_n_steps=int(train_cfg.get("log_every_n_steps", 50)),
        val_check_interval=int(train_cfg.get("val_check_interval", 500)),
        fast_dev_run=args.fast_dev_run,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return 0


def _required_path(cfg: dict[str, Any], section: str, key: str, root: Path) -> Path:
    value = cfg.get(section, {}).get(key)
    if value is None:
        raise KeyError(f"Missing config value: {section}.{key}")
    path = resolve_path(value, root)
    if path is None:
        raise KeyError(f"Missing config value: {section}.{key}")
    return path


def _apply_hamer_overrides(model_cfg: Any, cfg: dict[str, Any]) -> None:
    train_cfg = cfg.get("train", {})
    loss_weights = cfg.get("loss_weights", {})
    model_cfg.defrost()
    model_cfg.TRAIN.LR = float(train_cfg.get("lr", model_cfg.TRAIN.get("LR", 1.0e-5)))
    model_cfg.TRAIN.WEIGHT_DECAY = float(train_cfg.get("weight_decay", model_cfg.TRAIN.get("WEIGHT_DECAY", 1.0e-4)))
    model_cfg.TRAIN.BATCH_SIZE = int(train_cfg.get("batch_size", model_cfg.TRAIN.get("BATCH_SIZE", 8)))
    model_cfg.GENERAL.NUM_WORKERS = int(train_cfg.get("num_workers", model_cfg.GENERAL.get("NUM_WORKERS", 4)))
    model_cfg.GENERAL.PREFETCH_FACTOR = int(train_cfg.get("prefetch_factor", model_cfg.GENERAL.get("PREFETCH_FACTOR", 2)))
    for key, value in loss_weights.items():
        model_cfg.LOSS_WEIGHTS[key] = float(value)
    model_cfg.LOSS_WEIGHTS.ADVERSARIAL = 0.0
    if model_cfg.MODEL.BACKBONE.get("TYPE") == "vit" and "BBOX_SHAPE" not in model_cfg.MODEL:
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
    if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
        model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
    model_cfg.freeze()


if __name__ == "__main__":
    raise SystemExit(main())

