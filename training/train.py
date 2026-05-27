"""Main training script for DDPM."""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.unet import UNet
from training.diffusion import GaussianDiffusion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_dataset(cfg):
    transform = transforms.Compose([
        transforms.Resize(cfg.model.image_size),
        transforms.CenterCrop(cfg.model.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    if cfg.data.dataset == "cifar10":
        return datasets.CIFAR10(cfg.data.data_dir, train=True, download=True, transform=transform)
    elif cfg.data.dataset == "celeba":
        return datasets.CelebA(cfg.data.data_dir, split="train", download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.dataset}")


def train(cfg):
    accelerator = Accelerator(
        mixed_precision=cfg.training.mixed_precision,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        log_with="wandb",
        project_dir=cfg.output.dir,
    )
    set_seed(42)

    if accelerator.is_main_process:
        os.makedirs(cfg.output.checkpoints, exist_ok=True)
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": {"name": cfg.logging.run_name}},
        )

    dataset = get_dataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
    )

    model = UNet(
        image_size=cfg.model.image_size,
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        base_channels=cfg.model.base_channels,
        channel_mults=list(cfg.model.channel_mults),
        num_res_blocks=cfg.model.num_res_blocks,
        attention_resolutions=list(cfg.model.attention_resolutions),
        dropout=cfg.model.dropout,
    )

    diffusion = GaussianDiffusion(
        timesteps=cfg.diffusion.timesteps,
        beta_schedule=cfg.diffusion.beta_schedule,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        prediction_type=cfg.diffusion.prediction_type,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.training.lr,
        total_steps=cfg.training.num_epochs * len(loader),
        pct_start=cfg.training.lr_warmup_steps / (cfg.training.num_epochs * len(loader)),
    )

    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)

    global_step = 0
    for epoch in range(cfg.training.num_epochs):
        model.train()
        pbar = tqdm(loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}")
        for x, _ in pbar:
            with accelerator.accumulate(model):
                t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=x.device)
                loss = diffusion.training_loss(accelerator.unwrap_model(model), x, t)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            pbar.set_postfix(loss=loss.item())

            if global_step % cfg.training.log_every == 0 and accelerator.is_main_process:
                accelerator.log({"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0]}, step=global_step)

        if (epoch + 1) % cfg.training.save_every == 0 and accelerator.is_main_process:
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "model": accelerator.unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(ckpt, f"{cfg.output.checkpoints}/epoch_{epoch+1:04d}.pt")
            logger.info(f"Saved checkpoint at epoch {epoch+1}")

        if (epoch + 1) % (cfg.training.eval_every // len(loader) or 1) == 0 and accelerator.is_main_process:
            model.eval()
            samples = diffusion.sample(
                accelerator.unwrap_model(model),
                shape=(min(16, cfg.training.num_eval_samples), cfg.model.in_channels, cfg.model.image_size, cfg.model.image_size),
                device=accelerator.device,
            )
            grid = make_grid((samples * 0.5 + 0.5).clamp(0, 1), nrow=4)
            save_image(grid, f"{cfg.output.dir}/samples_epoch_{epoch+1:04d}.png")
            accelerator.log({"samples": wandb.Image(grid.cpu().numpy().transpose(1, 2, 0))}, step=global_step)

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_ddpm.yaml")
    args, overrides = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    train(cfg)
