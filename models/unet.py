"""UNet backbone for DDPM."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
    ).to(t.device)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch * 2)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_proj(F.silu(t_emb)).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = rearrange(h, 'b c h w -> b (h w) c')
        h, _ = self.attn(h, h, h)
        h = rearrange(h, 'b (h w) c -> b c h w', h=H, w=W)
        return x + h


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))


class UNet(nn.Module):
    def __init__(
        self,
        image_size=64,
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        dropout=0.0,
    ):
        super().__init__()
        time_emb_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        ch = base_channels
        self.input_conv = nn.Conv2d(in_channels, ch, 3, padding=1)

        # encoder
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        skips = [ch]
        cur_res = image_size
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(ch, out_ch, time_emb_dim, dropout))
                if cur_res in attention_resolutions:
                    self.down_blocks.append(AttentionBlock(out_ch))
                else:
                    self.down_blocks.append(nn.Identity())
                skips.append(out_ch)
                ch = out_ch
            if i < len(channel_mults) - 1:
                self.down_samples.append(Downsample(ch))
                skips.append(ch)
                cur_res //= 2
            else:
                self.down_samples.append(nn.Identity())

        # bottleneck
        self.mid_res1 = ResBlock(ch, ch, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid_res2 = ResBlock(ch, ch, time_emb_dim, dropout)

        # decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for j in range(num_res_blocks + 1):
                skip_ch = skips.pop()
                self.up_blocks.append(ResBlock(ch + skip_ch, out_ch, time_emb_dim, dropout))
                if cur_res in attention_resolutions:
                    self.up_blocks.append(AttentionBlock(out_ch))
                else:
                    self.up_blocks.append(nn.Identity())
                ch = out_ch
            if i > 0:
                self.up_samples.append(Upsample(ch))
                cur_res *= 2
            else:
                self.up_samples.append(nn.Identity())

        self.num_res_blocks = num_res_blocks
        self.output_norm = nn.GroupNorm(32, ch)
        self.output_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = timestep_embedding(t, self.time_embed[0].in_features)
        t_emb = self.time_embed(t_emb)

        h = self.input_conv(x)
        hs = [h]

        # encode
        block_idx = 0
        for ds in self.down_samples:
            for _ in range(self.num_res_blocks):
                res, attn = self.down_blocks[block_idx], self.down_blocks[block_idx + 1]
                block_idx += 2
                h = res(h, t_emb) if isinstance(res, ResBlock) else res(h)
                h = attn(h) if isinstance(attn, AttentionBlock) else h
                hs.append(h)
            h = ds(h)
            if not isinstance(ds, nn.Identity):
                hs.append(h)

        # bottleneck
        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb)

        # decode
        block_idx = 0
        for us in self.up_samples:
            for _ in range(self.num_res_blocks + 1):
                res, attn = self.up_blocks[block_idx], self.up_blocks[block_idx + 1]
                block_idx += 2
                h = torch.cat([h, hs.pop()], dim=1)
                h = res(h, t_emb) if isinstance(res, ResBlock) else res(h)
                h = attn(h) if isinstance(attn, AttentionBlock) else h
            h = us(h)

        return self.output_conv(F.silu(self.output_norm(h)))
