from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from hamer.models.hamer import HAMER


class ArticHAMERFineTuner(HAMER):
    """HaMeR fine-tuner for supervised ARTIC batches without mocap adversarial loss."""

    def __init__(self, cfg: Any, freeze_backbone: bool = True, init_renderer: bool = False) -> None:
        super().__init__(cfg, init_renderer=init_renderer)
        self.freeze_backbone = freeze_backbone
        self.automatic_optimization = False
        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            self.backbone.eval()

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def get_parameters(self):  # type: ignore[override]
        if self.freeze_backbone:
            return list(self.mano_head.parameters())
        return super().get_parameters()

    def configure_optimizers(self):  # type: ignore[override]
        return torch.optim.AdamW(
            params=filter(lambda parameter: parameter.requires_grad, self.get_parameters()),
            lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int):  # type: ignore[override]
        del batch_idx
        optimizer = self.optimizers(use_pl_optimizer=True)
        output = self.forward_step(batch, train=True)
        loss = self.compute_loss(batch, output, train=True)
        if torch.isnan(loss):
            raise ValueError("Loss is NaN")
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        self._log_losses("train", output, batch["img"].shape[0])
        return {"loss": loss.detach()}

    def validation_step(self, batch: dict[str, Any], batch_idx: int):  # type: ignore[override]
        del batch_idx
        output = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, output, train=False)
        self._log_losses("val", output, batch["img"].shape[0])
        return {"loss": loss.detach()}

    def _log_losses(self, stage: str, output: dict[str, Any], batch_size: int) -> None:
        for name, value in output.get("losses", {}).items():
            self.log(
                f"{stage}/{name}",
                value,
                batch_size=batch_size,
                on_step=stage == "train",
                on_epoch=stage != "train",
                prog_bar=name == "loss",
                sync_dist=True,
            )


def load_hamer_weights(model: ArticHAMERFineTuner, checkpoint_path: str | Path) -> dict[str, list[str]]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return {"missing": list(missing), "unexpected": list(unexpected)}

