"""DDPM noise schedule and loss."""

import torch
import numpy as np


def make_beta_schedule(schedule, timesteps, beta_start, beta_end):
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, timesteps)
    elif schedule == "cosine":
        s = 0.008
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps) / timesteps
        alpha_bar = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return betas.clamp(0, 0.999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_schedule="linear", beta_start=1e-4, beta_end=2e-2, prediction_type="epsilon"):
        self.timesteps = timesteps
        self.prediction_type = prediction_type

        betas = make_beta_schedule(beta_schedule, timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register = lambda name, val: setattr(self, name, val)
        for name, val in [
            ("betas", betas),
            ("alphas_cumprod", alphas_cumprod),
            ("alphas_cumprod_prev", alphas_cumprod_prev),
            ("sqrt_alphas_cumprod", alphas_cumprod.sqrt()),
            ("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt()),
            ("log_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).log()),
            ("sqrt_recip_alphas_cumprod", (1.0 / alphas_cumprod).sqrt()),
            ("sqrt_recipm1_alphas_cumprod", (1.0 / alphas_cumprod - 1).sqrt()),
            ("posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        ]:
            self.register(name, val)

    def _extract(self, a, t, shape):
        b = t.shape[0]
        out = a.to(t.device).gather(0, t)
        return out.reshape(b, *((1,) * (len(shape) - 1)))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_alpha * x0 + sqrt_one_minus * noise, noise

    def training_loss(self, model, x0, t):
        xt, noise = self.q_sample(x0, t)
        pred = model(xt, t)
        if self.prediction_type == "epsilon":
            return torch.nn.functional.mse_loss(pred, noise)
        elif self.prediction_type == "x0":
            return torch.nn.functional.mse_loss(pred, x0)
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

    @torch.no_grad()
    def p_sample(self, model, x, t_idx):
        t = torch.full((x.shape[0],), t_idx, device=x.device, dtype=torch.long)
        pred = model(x, t)

        if self.prediction_type == "epsilon":
            sqrt_recip = self._extract(self.sqrt_recip_alphas_cumprod, t, x.shape)
            sqrt_recipm1 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
            x0_pred = sqrt_recip * x - sqrt_recipm1 * pred
        else:
            x0_pred = pred
        x0_pred = x0_pred.clamp(-1, 1)

        posterior_mean = (
            self._extract(self.betas, t, x.shape) * self._extract(self.alphas_cumprod_prev, t, x.shape).sqrt() / (1.0 - self._extract(self.alphas_cumprod, t, x.shape)) * x0_pred
            + self._extract((1.0 - self.alphas_cumprod_prev) * (1.0 - self.betas).sqrt(), t, x.shape) / (1.0 - self._extract(self.alphas_cumprod, t, x.shape)) * x
        )
        posterior_var = self._extract(self.posterior_variance, t, x.shape)

        noise = torch.randn_like(x) if t_idx > 0 else torch.zeros_like(x)
        return posterior_mean + posterior_var.sqrt() * noise

    @torch.no_grad()
    def sample(self, model, shape, device):
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(model, x, t)
        return x
