# HOHS MANO Regressor

Project scaffold for fine-tuning the HaMeR MANO regressor on ARTIC hand data with true 3D supervision. HaMeR stays as an editable third-party checkout under `third_party/hamer`; this repo holds our ARTIC conversion, fine-tuning wrapper, configs, and server bootstrap scripts.

## Current Backbone

- Upstream backbone: `geopavlakos/hamer`
- Target dataset: ARTIC converted into HaMeR-compatible NPZ annotations
- Default fine-tune mode: load the released HaMeR checkpoint, freeze the ViT backbone, and train the MANO transformer decoder/head with 3D keypoint, 2D keypoint, and MANO parameter losses

## Local/Server Setup

On the GPU server, from this repo:

```bash
bash scripts/bootstrap_hamer.sh
```

The script creates a venv under `/data/hohs2/venvs/hohs_mano_regressor` on AIT, loads `gcc/11 cuda/11.8 cudnn/8.4_cuda11.x` by default, stores pip cache under `/data/hohs2/pip-cache`, clones HaMeR recursively, installs CUDA PyTorch, installs HaMeR editably, installs this package, and downloads HaMeR checkpoint assets.

Detectron2, ViTPose/MMPose, and mmcv are skipped by default because ARTIC fine-tuning uses annotated crops and does not need demo-time hand/body localization. To attempt a full demo-capable install, set `INSTALL_DETECTRON2=1 INSTALL_VITPOSE=1`.

If `python3.10` is not available on the server, first install a local CPython runtime under `/data/hohs2/python/`:

```bash
bash scripts/install_python310.sh
```

HaMeR also requires the licensed MANO right-hand model. Put `MANO_RIGHT.pkl` at:

```text
third_party/hamer/_DATA/data/mano/MANO_RIGHT.pkl
```

## Prepare ARTIC

Convert ARTIC annotations into the NPZ fields expected by HaMeR's `ImageDataset`:

```bash
python scripts/prepare_artic_npz.py \
  --source data/artic/raw/train_annotations.npz \
  --output data/artic/processed/train_hamer.npz \
  --image-root data/artic/images
```

Repeat for validation. The converter accepts NPZ, JSON, or JSONL manifests with common aliases such as `image_path`, `hand_keypoints_2d`, `hand_keypoints_3d`, `bbox_xyxy`, `global_orient`, `hand_pose`, and `betas`.

## Train

After the environment, checkpoint, MANO file, images, and converted NPZ files are in place:

```bash
bash scripts/train_artic.sh
```

Override config values from the CLI by editing `configs/train_artic.yaml` for now. Outputs are written to `outputs/artic_hamer`. The launch script defaults to one visible GPU through `CUDA_VISIBLE_DEVICES=0`; change that only after checking `nvidia-smi` on the shared server.

## Smoke Check

```bash
python scripts/smoke_test.py --config configs/train_artic.yaml
```

Use `--load-model` on the GPU server once the HaMeR checkpoint and MANO assets are available.
