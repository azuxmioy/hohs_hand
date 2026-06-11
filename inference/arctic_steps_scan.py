"""Ablation over denoise steps at fixed guidance=30 for the ARCTIC inpainting.

Loads the model once and runs chosen samples across several step counts (same
initial noise per sample), writing one comparison grid:
    row = sample;  cols = original | masked | condition | gen@steps0 | gen@steps1 | ...

Run in the `hohs_hand` env:
    python inference/arctic_steps_scan.py \
        --checkpoint /data/.../step_008000 \
        --h5 /data/hohs2/arctic/laptop_use_01_ego_sam.h5 \
        --embeddings /data/.../text_embeddings.pt \
        --out /data/hohs2/arctic/steps_scan.png \
        --indices 4,5 --steps 15,30,50,75 --guidance 30
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL, FluxControlNetModel, FluxTransformer2DModel
from omegaconf import OmegaConf
from peft import LoraConfig
from peft.utils import set_peft_model_state_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.hand_dataset import HandDataset
from inference.arctic_inpaint import to_uint8
from inference.arctic_cfg_scan import denoise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_flux_a100_lora.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--indices", default="4,5")
    ap.add_argument("--steps", default="15,30,50,75")
    ap.add_argument("--guidance", type=float, default=30.0)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    dtype, device = torch.bfloat16, "cuda"
    indices = [int(x) for x in args.indices.split(",")]
    steps_list = [int(x) for x in args.steps.split(",")]

    print("Loading VAE + transformer …")
    vae = AutoencoderKL.from_pretrained(cfg.model.base_model, subfolder="vae", torch_dtype=dtype).to(device)
    transformer = FluxTransformer2DModel.from_pretrained(cfg.model.base_model, subfolder="transformer", torch_dtype=dtype).to(device)
    vae.requires_grad_(False); transformer.requires_grad_(False)
    lc = cfg.training.lora
    transformer.add_adapter(LoraConfig(r=lc.rank, lora_alpha=lc.alpha,
                            init_lora_weights="gaussian", target_modules=list(lc.target_modules)))
    set_peft_model_state_dict(transformer, torch.load(Path(args.checkpoint) / "transformer_lora.pt", map_location="cpu"))
    transformer.to(dtype=dtype)
    controlnet = FluxControlNetModel.from_pretrained(Path(args.checkpoint) / "controlnet", torch_dtype=dtype).to(device)
    transformer.eval(); controlnet.eval()

    emb = torch.load(args.embeddings, map_location="cpu", weights_only=True)
    pe, ppe = emb["prompt_embeds"].to(dtype=dtype), emb["pooled_prompt_embeds"].to(dtype=dtype)

    ds = HandDataset(args.h5, image_size=args.image_size, augment=False,
                     use_stored_skeleton=True, mask_dilation_max=0)

    rows = []
    for idx in indices:
        batch = ds[idx]
        orig = masked = cond = None
        gens = []
        for ns in steps_list:
            o, m, c, gen = denoise(transformer, vae, controlnet, batch, pe, ppe,
                                   device, dtype, args.image_size, ns,
                                   args.guidance, seed=args.seed + idx)
            orig, masked, cond = o, m, c
            gens.append(gen)
            print(f"  idx {idx}  steps={ns} done")
        panels = [to_uint8(orig), to_uint8(masked), to_uint8(cond)] + [to_uint8(g) for g in gens]
        rows.append(np.concatenate(panels, axis=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(args.out)
    print(f"Wrote {args.out}  cols: original | masked | condition | " +
          " | ".join(f"s{n}" for n in steps_list))


if __name__ == "__main__":
    main()
