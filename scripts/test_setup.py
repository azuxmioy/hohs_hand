"""Quick smoke test — runs one training step to verify the env is correct."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
print(f"torch version : {torch.__version__}")
print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

from models.unet import UNet
from training.diffusion import GaussianDiffusion

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(image_size=32, base_channels=64, channel_mults=(1, 2, 4), num_res_blocks=1).to(device)
diffusion = GaussianDiffusion(timesteps=100)

x = torch.randn(2, 3, 32, 32, device=device)
t = torch.randint(0, 100, (2,), device=device)
loss = diffusion.training_loss(model, x, t)
print(f"Smoke-test loss : {loss.item():.4f}  [OK]")

sample = diffusion.sample(model, (2, 3, 32, 32), device)
print(f"Sample shape   : {sample.shape}  [OK]")
print("All checks passed.")
