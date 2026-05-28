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
    parser.add_argument("--resolution", type=int, default=512,
                        help="Run inpainting at this resolution. FLUX is trained at 1024.")
    parser.add_argument("--rope_interp", action="store_true",
                        help="Scale RoPE image_ids by (1024/resolution) so the model "
                             "sees the same positional range it was trained on.")
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

    if args.rope_interp:
        # Monkey-patch _prepare_latent_image_ids to scale by 1024/resolution.
        # This puts the position IDs in the same numeric range the model saw
        # during 1024×1024 training even when we run at lower resolution.
        scale = 1024.0 / args.resolution

        @staticmethod
        def _prepare_latent_image_ids_scaled(batch_size, height, width, device, dtype):
            ids = torch.zeros(height, width, 3)
            ids[..., 1] = torch.arange(height)[:, None] * scale
            ids[..., 2] = torch.arange(width)[None, :] * scale
            return ids.reshape(height * width, 3).to(device=device, dtype=dtype)

        FluxFillPipeline._prepare_latent_image_ids = _prepare_latent_image_ids_scaled
        print(f"RoPE position interpolation enabled (scale={scale:.2f})")

    # Save panels for each sample: [original | masked | inpainted]
    R = args.resolution
    with h5py.File(args.hdf5_path, "r") as f:
        for i in range(args.num_samples):
            idx = args.sample_idx + i
            crop = f["crops"][idx]              # (256,256,3) uint8
            mask = (f["masks"][idx] > 0).astype(np.uint8) * 255  # binary 0/255

            image_pil = Image.fromarray(crop).resize((R, R), Image.BILINEAR)
            mask_pil = Image.fromarray(mask).resize((R, R), Image.NEAREST)
            mask_arr = np.array(mask_pil)
            masked_pil = Image.fromarray(np.array(image_pil) * (mask_arr[..., None] == 0))

            generator = torch.Generator(device=device).manual_seed(42)
            result = pipe(
                prompt=args.prompt,
                image=image_pil,
                mask_image=mask_pil,
                height=R,
                width=R,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]

            panel = Image.new("RGB", (R * 3, R))
            panel.paste(image_pil, (0, 0))
            panel.paste(masked_pil, (R, 0))
            panel.paste(result, (R * 2, 0))
            suffix = f"_r{R}" + ("_ropeinterp" if args.rope_interp else "")
            out_path = os.path.join(args.out_dir, f"sample_{idx:04d}{suffix}.png")
            panel.save(out_path)
            print(f"Saved {out_path}")

    print(f"\nDone. Panels: original | masked | inpainted (prompt: {args.prompt!r})")


if __name__ == "__main__":
    main()
