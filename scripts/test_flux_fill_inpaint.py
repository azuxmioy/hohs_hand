"""
Quick text-guided inpainting smoke test.

Uses the official FluxFillPipeline (no ControlNet) with a real text prompt
to verify the base model can inpaint hands when given proper text guidance.
If this works, it confirms the random-output issue at step 0 of our training
is caused by empty prompt + zero-init'd ControlNet (both expected).

Usage:
    python scripts/test_flux_fill_inpaint.py \\
        --sample_idx 0 \\
        --prompt "a human hand"
"""
import argparse
import os
import h5py
import numpy as np
import torch
from PIL import Image
from diffusers import FluxFillPipeline, FluxTransformer2DModel
from transformers import BitsAndBytesConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_path", default="/data/hohs2/datasets/data.h5")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--prompt", default="a human hand")
    parser.add_argument(
        "--out_dir",
        default="/data/hohs2/outputs/flux_controlnet/text_guided_test",
    )
    parser.add_argument("--num_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=30.0)
    parser.add_argument("--num_samples", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    dtype = torch.bfloat16

    # NF4-quantized transformer to fit in 24 GB
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev",
        subfolder="transformer",
        quantization_config=bnb_config,
        torch_dtype=dtype,
    )
    pipe = FluxFillPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev",
        transformer=transformer,
        torch_dtype=dtype,
    )
    pipe.to(device)

    # Save panels for each sample: [original | masked | inpainted]
    with h5py.File(args.hdf5_path, "r") as f:
        for i in range(args.num_samples):
            idx = args.sample_idx + i
            crop = f["crops"][idx]              # (256,256,3) uint8
            mask = (f["masks"][idx] > 0).astype(np.uint8) * 255  # binary 0/255

            image_pil = Image.fromarray(crop)
            mask_pil = Image.fromarray(mask)
            masked_pil = Image.fromarray(crop * (mask[..., None] == 0))

            generator = torch.Generator(device=device).manual_seed(42)
            result = pipe(
                prompt=args.prompt,
                image=image_pil,
                mask_image=mask_pil,
                height=256,
                width=256,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]

            panel = Image.new("RGB", (256 * 3, 256))
            panel.paste(image_pil, (0, 0))
            panel.paste(masked_pil, (256, 0))
            panel.paste(result, (512, 0))
            out_path = os.path.join(args.out_dir, f"sample_{idx:04d}.png")
            panel.save(out_path)
            print(f"Saved {out_path}")

    print(f"\nDone. Panels: original | masked | inpainted (prompt: {args.prompt!r})")


if __name__ == "__main__":
    main()
