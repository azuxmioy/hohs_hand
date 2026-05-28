# hohs_hand — FLUX.1-Fill + ControlNet on 24 GB GPUs

A training pipeline for **hand inpainting** that fine-tunes a small
ControlNet on top of a frozen **FLUX.1-Fill-dev** transformer, using
skeleton + MANO-UV as conditioning. The repo is designed to fit on a
single 24 GB GPU (Quadro RTX 6000) and scale to multi-GPU via Accelerate.

```
masked image  +  hand skeleton  +  MANO UV map   →   inpainted hand
```

## Why this is interesting

FLUX.1-Fill-dev is a 12 B-parameter rectified-flow transformer. Out of
the box, training a ControlNet alongside it would not fit on a 24 GB
card. This repo combines three independent memory tricks so the whole
thing — base model, ControlNet, optimizer, activations — lives inside
24 GB at 512 × 512 resolution:

| Optimization | Savings | What it does |
|---|---|---|
| **NF4 4-bit quantization** of the frozen transformer | ~18 GB | bitsandbytes `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` |
| **Gradient checkpointing** on transformer + ControlNet | ~10-15 GB | Re-computes activations during backward |
| **8-bit AdamW** for the trainable ControlNet | ~8 GB | `bitsandbytes.optim.AdamW8bit` |

Plus pre-computed VAE latents and text embeddings so the only thing the
training step touches is the 12 B transformer (NF4) and the small
ControlNet. Result: 4-GPU DDP training at 512 × 512 fits in **~19 GB
per card** with ~11.5 s per step.

## Repo layout

```
configs/
  train_flux.yaml             256-px config (early experiments)
  train_flux_512.yaml         512-px config (recommended)
data/
  hand_dataset.py             Raw HDF5 dataset (used by 256-px training)
  latent_dataset.py           Cached-latents dataset with rot90 augmentation
training/
  train_flux_controlnet.py            256-px training
  train_flux_controlnet_latent.py     512-px training (latent cache + 8-bit AdamW)
scripts/
  precompute_flux_embeddings.py       Pre-compute T5+CLIP empty-prompt embeddings
  precompute_latents.py               VAE-encode all samples into an HDF5 cache
  launch_flux_train.sh                Launcher for 256-px training
  launch_flux_train_512.sh            Launcher for 512-px training (precompute + train)
  test_flux_fill_inpaint.py           Standalone inpainting smoke test
```

## Training at 512 × 512 (recommended)

FLUX.1-Fill was trained at 1024 × 1024; running it at 256 produces
incoherent output because the model is far outside its training
distribution. 512 is the practical sweet spot for 24 GB GPUs.

```bash
# 1. Place data.h5 under whatever you want $DATA_DIR/datasets/ to be.
#    Default $DATA_DIR is /data/$USER; override via env if you need to.
#       export DATA_DIR=/path/to/your/data_dir
# 2. Launch — picks 4 free GPUs, runs both precomputes if needed,
#    then trains with accelerate launch --multi_gpu
bash scripts/launch_flux_train_512.sh --num-gpus 4
```

What the launcher does:

1. Picks N GPUs whose utilization is 0 % and memory < 100 MiB
2. Builds the text-embedding cache (empty prompt, T5 + CLIP) — once
3. Builds the VAE-latent cache (`data_latents_512.h5`, ~10 GB) — once
4. Launches `accelerate launch --multi_gpu --num_processes=N`
5. Exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and a
   30-minute NCCL collective timeout so per-process eval/save don't
   trip the default 10-minute watchdog

## Conditioning

The ControlNet sees a single RGB composite of **skeleton lines drawn
over the MANO UV map** (see `data/hand_dataset.py:render_skeleton` and
the `skeleton_on_uv` composite). This gives both pose (21 keypoints,
finger-coloured) and texture-orientation guidance in one image. The
frozen FLUX-Fill backbone simultaneously sees the masked image and
binary mask through its native 384-channel inpainting input.

## Dataset

The repo expects an HDF5 file with these datasets (any spatial size,
resized in the loader):

| Key | Shape | dtype | Notes |
|---|---|---|---|
| `crops` | `(N, H, W, 3)` | uint8 | RGB hand crops |
| `masks` | `(N, H, W)` | uint8 | 0 / 255, the hand region |
| `skeletons` | `(N, H, W, 3)` | uint8 | Pre-rendered 21-keypoint skeleton on black bg |
| `uvs` | `(N, H, W, 3)` | uint8 | MANO UV map (RGB encoding of (u, v)) |
| `keypoints_2d_output` | `(N, 21, 2)` | float32 | Pixel coordinates |
| `keypoints_3d` | `(N, 21, 3)` | float32 | Metric coordinates |
| `is_right` | `(N,)` | bool | Hand side |

Augmentation is **random 90 ° rotation** (no flip, since flip changes
left/right hand identity); for the 512 pipeline this is applied in
latent space after the cache is built.

## Hardware notes

Verified on Quadro RTX 6000 (24 GB). Should also work on RTX 3090 /
4090, A5000, A4500. The NF4 quantization needs a GPU that supports
bitsandbytes 4-bit (compute capability ≥ 7.5).

## Acknowledgments

Built with:
- [diffusers](https://github.com/huggingface/diffusers) (`FluxFillPipeline`, `FluxControlNetModel`)
- [accelerate](https://github.com/huggingface/accelerate) (DDP + NCCL kwargs)
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) (NF4 quantization, 8-bit AdamW)
- [Black Forest Labs FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)
