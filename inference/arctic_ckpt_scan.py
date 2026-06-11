"""Ablation over training checkpoints (step_002000..008000) for the ARCTIC inpainting.

Loads the frozen base (VAE + transformer) once, then for each checkpoint swaps in its
transformer-LoRA weights + ControlNet and runs the chosen samples (same noise seed),
writing one comparison grid:
    row = sample;  cols = original | masked | condition | gen@2000 | gen@4000 | ...

Run in the `hohs_hand` env:
    python inference/arctic_ckpt_scan.py \
        --run-dir /data/hohs2/checkpoints/flux_controlnet_lora/20260529_211730 \
        --steps 2000,4000,6000,8000 \
        --h5 /data/hohs2/arctic/laptop_use_01_ego_sam.h5 \
        --embeddings /data/.../text_embeddings.pt \
        --out /data/hohs2/arctic/ckpt_scan.png --indices 4,5,0,1
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
    ap.add_argument("--run-dir", required=True, help="dir containing step_XXXXXX subdirs")
    ap.add_argument("--steps", default="2000,4000,6000,8000")
    ap.add_argument("--h5", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--indices", default="4,5,0,1")
    ap.add_argument("--guidance", type=float, default=30.0)
    ap.add_argument("--num-steps", type=int, default=30)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    dtype, device = torch.bfloat16, "cuda"
    indices = [int(x) for x in args.indices.split(",")]
    ckpt_steps = [int(x) for x in args.steps.split(",")]
    run_dir = Path(args.run_dir)

    print("Loading VAE + base transformer …")
    vae = AutoencoderKL.from_pretrained(cfg.model.base_model, subfolder="vae", torch_dtype=dtype).to(device)
    transformer = FluxTransformer2DModel.from_pretrained(cfg.model.base_model, subfolder="transformer", torch_dtype=dtype).to(device)
    vae.requires_grad_(False); transformer.requires_grad_(False)
    lc = cfg.training.lora
    transformer.add_adapter(LoraConfig(r=lc.rank, lora_alpha=lc.alpha,
                            init_lora_weights="gaussian", target_modules=list(lc.target_modules)))
    transformer.eval()

    emb = torch.load(args.embeddings, map_location="cpu", weights_only=True)
    pe, ppe = emb["prompt_embeds"].to(dtype=dtype), emb["pooled_prompt_embeds"].to(dtype=dtype)
    ds = HandDataset(args.h5, image_size=args.image_size, augment=False,
                     use_stored_skeleton=True, mask_dilation_max=0)

    # Run per checkpoint, caching each sample's panels.
    panels = {idx: [to_uint8(ds[idx]["image"]),
                    to_uint8(ds[idx]["masked_image"]),
                    to_uint8(ds[idx]["condition"])] for idx in indices}
    for st in ckpt_steps:
        ckpt = run_dir / f"step_{st:06d}"
        print(f"== checkpoint {ckpt.name} ==")
        set_peft_model_state_dict(transformer, torch.load(ckpt / "transformer_lora.pt", map_location="cpu"))
        transformer.to(dtype=dtype)
        controlnet = FluxControlNetModel.from_pretrained(ckpt / "controlnet", torch_dtype=dtype).to(device)
        controlnet.eval()
        for idx in indices:
            _, _, _, gen = denoise(transformer, vae, controlnet, ds[idx], pe, ppe,
                                   device, dtype, args.image_size, args.num_steps,
                                   args.guidance, seed=args.seed + idx)
            panels[idx].append(to_uint8(gen))
            print(f"  idx {idx} done")
        del controlnet
        torch.cuda.empty_cache()

    rows = [np.concatenate(panels[idx], axis=1) for idx in indices]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(args.out)
    print(f"Wrote {args.out}  cols: original | masked | condition | " +
          " | ".join(f"step{st}" for st in ckpt_steps))


if __name__ == "__main__":
    main()
