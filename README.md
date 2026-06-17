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
  generate_hand_crops.py              Build the conditioning HDF5 from MANO + masks
inference/
  arctic_make_conditions.py           ARCTIC GT-MANO + camera -> conditioning HDF5
  palm_make_conditions.py             PALM (cameras.npy + poses.npy) -> conditioning HDF5
  sam2_masks.py                       Occlusion-aware SAM2 hand masks (wrist prompt)
  arctic_inpaint.py                   Inpaint with a trained ControlNet+LoRA checkpoint
  arctic_{cfg,steps,ckpt}_scan.py     Guidance / steps / checkpoint ablations
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

## Data preparation

`scripts/generate_hand_crops.py` builds the conditioning HDF5 from MANO results +
masks: for each frame/hand it computes a square crop from the 2D keypoints
(`--scale`), renders the 21-keypoint skeleton and the MANO-mesh UV map, crops the
tight hand mask, and writes the `data.h5` keys above. The training set was made
with `--scale 3 --out-size 512` (the crop is 3× the keypoint bbox, so the hand sits
small with surrounding context — match this scale when preparing new data).

## Reproduce: training (separate train/val sets)

By default a single `hdf5_path` is split into train/val. To train on one set and
**validate on a different distribution**, set `data.train_hdf5` + `data.val_hdf5`
in the config — `configs/train_flux_a100_lora_calib.yaml` trains on a new
(markerless-glove) set and validates on a prepared ARCTIC sequence. Implemented by
`data/hand_dataset.py:make_train_val_loaders`; each run writes a **timestamped**
checkpoint dir (`<checkpoints>/<run_id>/step_XXXXXX/{controlnet, transformer_lora.pt}`)
and logs `val/samples` to wandb.

```bash
# 1. Prepare the HDF5s (see Data preparation above): a train set and a held-out
#    validation set. Point the config at them:
#      data.train_hdf5: .../calib_data.h5            # what you train on
#      data.val_hdf5:   .../laptop_use_01_ego_sam.h5 # held-out ARCTIC (built below)

# 2. Launch on a shared server (single GPU). Run inside your own SSH session.
source /data/$USER/envs/hohs_hand/bin/activate     # use the venv, not system python
GPU=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
      | awk -F', ' '$2==0 && $3<100 {print $1; exit}')   # one idle GPU; run ONE job
HF_HOME=/data/$USER/cache/huggingface \
MALLOC_ARENA_MAX=2 TORCHINDUCTOR_COMPILE_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU \
  nohup python training/train_flux_controlnet_lora.py \
  --config configs/train_flux_a100_lora_calib.yaml > train.log 2>&1 &
```

Gotchas that matter on a shared host: `HF_HOME` **must** be set explicitly (a
non-interactive shell doesn't source `~/.bashrc`, so the gated FLUX repo would 401);
`MALLOC_ARENA_MAX` / `expandable_segments` keep the model load under the host's
memory-commit limit (`vm.overcommit_memory=2`); pick a free GPU and run **exactly one**
job (two on the same GPU → CUDA OOM). Warm-restart from a previous run's weights with
`--resume_from <run>/step_XXXXXX`.

The markerless-glove model converges by ~step 2000 and is stable thereafter; the
**chosen final checkpoint is `step_005200`** (compare iterations with
`inference/arctic_ckpt_scan.py`).

## Reproduce: inference on unseen data (ARCTIC)

Three steps turn an ARCTIC sequence into inpainted hands with a trained checkpoint.
Run in the `hohs_hand` env with `HF_HOME` set (as above).

```bash
SUBJ=s01; SEQ=laptop_use_01            # an egocentric, low-occlusion (hand-back-up) seq
CKPT=/data/$USER/checkpoints/flux_controlnet_lora/20260611_235529/step_005200  # final model
EMB=/data/$USER/outputs/flux_controlnet/text_embeddings.pt

# 1. ARCTIC GT-MANO + camera -> data.h5-compatible conditioning (egocentric, scale 3)
python inference/arctic_make_conditions.py \
  --arctic-root  /data/$USER/datasets/arctic_dl/extracted \
  --mano-dir     /data/$USER/datasets/arctic_dl/mano_v1_2/models \
  --uv-left  MANO_UV_left.obj --uv-right MANO_UV_right.obj \
  --subject $SUBJ --seq $SEQ --view 0 --scale 3 \
  --out cond_${SEQ}.h5

# 2. Occlusion-aware SAM2 hand mask (single wrist-point prompt), training-matched dilation
python inference/sam2_masks.py \
  --in-h5 cond_${SEQ}.h5 --out-h5 cond_${SEQ}_sam.h5 --mesh-pad 3 --dilate 8

# 3. Inpaint (guidance must stay ~30; 30 steps is plenty)
python inference/arctic_inpaint.py \
  --checkpoint $CKPT --h5 cond_${SEQ}_sam.h5 --embeddings $EMB \
  --out-dir results_${SEQ} --num-samples 8 --num-steps 30 --guidance 30
```

- `arctic_make_conditions.py` builds the MANO mesh via `smplx` and projects with
  `world2cam`/`intris_mat` (allocentric) or per-frame `world2ego` + `K_ego` + fisheye
  distortion (egocentric `--view 0`). `--scale 3` matches the training crop framing.
- `arctic_inpaint.py` loads ControlNet + transformer-LoRA and reuses
  `HandDataset(augment=False)`. **Keep `--guidance ≈ 30`** — FLUX is guidance-distilled
  and training pinned `guidance=30`, so other values collapse to noise.

### PALM (studio bare-hand capture)

`inference/palm_make_conditions.py` does the same for the PALM dataset (per subject:
`cameras.npy` with 7 cams `K`/`dist`/`Rt`, `poses.npy` with MANO `betas/global_orient/
hand_pose/transl`, plus `images/`, `masks/`, `mano/`). PALM is right-hand only and the
cameras are a fixed rig, so the front/dorsal views (e.g. `MCU_03`, `MCU_06`) frame the
hand best; side-on views (`MCU_01/07`) are foreshortened.

```bash
python inference/palm_make_conditions.py \
  --subj-dir /data/$USER/palm/<unzipped_subject_dir> \
  --mano-dir /data/$USER/datasets/arctic_dl/mano_v1_2/models \
  --uv-right MANO_UV_right.obj \
  --gesture 000001 --scale 3 --mask-dilate 8 --debug \
  --out cond_palm.h5
# then sam2_masks.py (optional) + arctic_inpaint.py, exactly as above.
```

### Ablation / comparison helpers

`inference/arctic_cfg_scan.py`, `arctic_steps_scan.py`, `arctic_ckpt_scan.py` sweep
guidance / steps / training-iteration on any conditioning h5 and write a **labeled**
grid (column headers, row ids, run/params caption). `arctic_ckpt_scan.py` takes
`--viz gen,blend,diff` (comma list): `gen` = raw inpaint, `blend` = 50% input + 50%
inpaint, `diff` = brightened `|inpaint − input|` (handy when the inpaint domain differs
from the input, e.g. glove model on bare-hand images). Example:

```bash
python inference/arctic_ckpt_scan.py \
  --run-dir <checkpoints>/<run_id> --steps 800,3600,5200 \
  --h5 cond_palm.h5 --embeddings <embeddings.pt> \
  --indices 2,3,5 --viz gen,diff --out ckpt_scan.png
```

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
