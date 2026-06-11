"""Quick guidance(CFG)/steps scan for the ARCTIC inpainting.

Loads the model once and runs a few chosen samples across several guidance values
(same initial noise per sample, so only CFG changes), writing one comparison grid:
    row = sample;  cols = original | masked | condition | gen@g0 | gen@g1 | ...

Run in the `hohs_hand` env:
    python inference/arctic_cfg_scan.py \
        --config configs/train_flux_a100_lora.yaml \
        --checkpoint /data/.../step_008000 \
        --h5 /data/hohs2/arctic/laptop_use_01_ego_sam.h5 \
        --embeddings /data/.../text_embeddings.pt \
        --out /data/hohs2/arctic/cfg_scan.png \
        --indices 4,5,0,1 --guidances 10,20,30,40,50 --num-steps 30
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
from inference.arctic_inpaint import (
    encode_images, pack_mask, unpack_latents, prepare_image_ids, to_uint8,
)


@torch.no_grad()
def denoise(transformer, vae, controlnet, batch, prompt_embeds, pooled_prompt_embeds,
            device, dtype, image_size, num_steps, guidance_scale, seed):
    img_ids = prepare_image_ids(image_size, image_size, device)
    txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=dtype)
    timesteps = torch.linspace(1.0, 1.0 / num_steps, num_steps, device=device, dtype=dtype)
    B = 1
    image        = batch["image"][None].to(device, dtype=dtype)
    masked_image = batch["masked_image"][None].to(device, dtype=dtype)
    condition    = batch["condition"][None].to(device, dtype=dtype)
    mask_binary  = batch["mask_binary"][None].to(device, dtype=dtype)

    masked_image_latents = encode_images(vae, masked_image)
    condition_latents    = encode_images(vae, condition)
    mask_packed          = pack_mask(mask_binary, vae_scale_factor=8)

    seq_len = (image_size // 16) ** 2
    g = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(B, seq_len, 64, device=device, dtype=dtype, generator=g)
    pe  = prompt_embeds.expand(B, -1, -1).to(device)
    ppe = pooled_prompt_embeds.expand(B, -1).to(device)
    guidance = torch.full((B,), guidance_scale, device=device, dtype=dtype)

    for t_val in timesteps:
        t_batch = torch.full((B,), t_val.item(), device=device, dtype=dtype)
        noisy = torch.cat([latents, masked_image_latents, mask_packed], dim=-1)
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            cn_b, cn_s = controlnet(
                hidden_states=latents, controlnet_cond=condition_latents,
                conditioning_scale=1.0, timestep=t_batch, guidance=guidance,
                encoder_hidden_states=pe, pooled_projections=ppe,
                img_ids=img_ids, txt_ids=txt_ids, return_dict=False)
            v = transformer(
                hidden_states=noisy, timestep=t_batch, guidance=guidance,
                encoder_hidden_states=pe, pooled_projections=ppe,
                img_ids=img_ids, txt_ids=txt_ids,
                controlnet_block_samples=cn_b, controlnet_single_block_samples=cn_s,
                return_dict=False)[0]
        latents = latents - (1.0 / num_steps) * v

    lat = unpack_latents(latents, image_size // 8, image_size // 8)
    lat = lat / vae.config.scaling_factor + vae.config.shift_factor
    gen = vae.decode(lat.to(dtype)).sample.clamp(-1, 1)
    return image[0].cpu().float(), masked_image[0].cpu().float(), \
        condition[0].cpu().float(), gen[0].cpu().float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_flux_a100_lora.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--indices", default="4,5,0,1")
    ap.add_argument("--guidances", default="10,20,30,40,50")
    ap.add_argument("--num-steps", type=int, default=30)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    dtype, device = torch.bfloat16, "cuda"
    indices = [int(x) for x in args.indices.split(",")]
    guidances = [float(x) for x in args.guidances.split(",")]

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
        for gscale in guidances:
            o, m, c, gen = denoise(transformer, vae, controlnet, batch, pe, ppe,
                                   device, dtype, args.image_size, args.num_steps,
                                   gscale, seed=args.seed + idx)
            orig, masked, cond = o, m, c
            gens.append(gen)
            print(f"  idx {idx}  g={gscale} done")
        panels = [to_uint8(orig), to_uint8(masked), to_uint8(cond)] + [to_uint8(g) for g in gens]
        rows.append(np.concatenate(panels, axis=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(args.out)
    print(f"Wrote {args.out}  cols: original | masked | condition | " +
          " | ".join(f"g{int(g)}" for g in guidances))


if __name__ == "__main__":
    main()
