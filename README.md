# SCV-MPDD

Generalizing **Structural Consistency Verification (SCV)** to general industrial anomaly
detection on the **MPDD** dataset.

## Background: the SCV paradigm

Our original method, **Structural Consistency Verification (SCV)**, is an unsupervised
anomaly-detection paradigm designed for **structural-absence anomalies** — missing
components that leave behind locally *normal-looking* background and therefore produce
almost no pixel-level residual. Instead of comparing pixels, SCV compares **instance
segmentation maps**: a region is masked and inpainted by a domain-adapted latent
diffusion model (a RealFill-style LoRA fine-tuning of Stable Diffusion 2 inpainting,
trained only on normal images); a fine-tuned segmentation model then extracts structural
instances from both the original and the inpainted image, and **mismatches between the
two segmentation maps (via IoU)** reveal anomalies at the semantic level. SCV was
developed and validated on a railway infrastructure inspection task (detecting missing
rail assets).

## This repository: generalizing SCV to MPDD

SCV as above presumes a domain with a well-defined structural vocabulary and a
segmentation model that captures it (rail assets + a fine-tuned FastSAM). Many industrial
inspection settings lack this: a benchmark such as **MPDD** spans heterogeneous object
categories with no single instance definition and no per-category segmentation
supervision.

To apply the framework in this regime, **the training procedure is kept unchanged** — a
RealFill-style LoRA is fine-tuned on the normal images of each category exactly as in
SCV, so the inpainter still learns that category's normal appearance. **Only the read-out
is replaced**: instead of comparing instance segmentations, we contrast the input with
its inpainted normal reconstruction in the **feature space of a frozen pretrained
network** (WideResNet-50-2). This preserves SCV's central idea — detecting anomalies by
comparing an image against a generatively-restored *normal* version of itself — while
removing the dependence on a domain-specific segmenter.

---

## Method (feature-difference read-out)

Given a test image (resized to `R x R`, default `256`):

1. **Full-coverage sliding-window reconstruction.**
   The image is split into a grid of `K x K` blocks (`K = R/8`, i.e. an 8×8 grid).
   A checkerboard schedule with stride `S=2` yields `S*S = 4` passes; each pass masks a
   quarter of non-adjacent blocks so the four passes tile the image **exactly once**.
   Every masked region is inpainted by the category's LoRA model and stitched back into a full reconstruction.
   This is repeated `N` times (default `10`) with different seeds and averaged — pure
   denoising that boosts the signal-to-noise ratio for small defects.

2. **Per-layer feature difference.**
   The original and reconstructed images are passed through an ImageNet-pretrained
   **WideResNet-50-2**; per-position squared L2 differences are taken at layers
   `l1`, `l2`, `l3`, upsampled to `R x R`, and Gaussian-blurred (σ=4).
   The `N` maps are averaged **at the difference-map level** (not by first averaging the
   reconstructions): reconstruction noise points in random directions across seeds and
   cancels, while the consistent defect signal survives.

3. **Fixed-scale normalization.**
   Each layer's diff map is divided by a single global scalar — that layer's mean diff
   averaged over the **normal ("good") test images only**. This balances the very
   different magnitudes across layers while preserving the cross-image signal (a
   defective image is globally brighter than a normal one) that image-level scoring
   depends on.

4. **Anomaly map.**  `M = D_l1' + D_l2' + D_l3'`  (all three layers).

5. **Scoring (both metrics from the same map `M`).**
   - **image-AUROC**: image score = mean of the top-1% brightest pixels of `M`.
   - **pixel-AUROC**: `M` compared pixel-wise against the ground-truth mask.

---

## 1. Requirements

- **OS**: Linux (tested), with an NVIDIA GPU (≈ 12 GB VRAM is enough at resolution 256).
- **CUDA**: 11.8 or 12.1 driver compatible with PyTorch 2.1.
- **Python**: 3.10.

Everything you need is in `requirements.txt`. The file pins a few transitive
dependencies as well as the top-level packages so that the tested
`diffusers==0.20.1` stack does not accidentally pick incompatible newer
releases.

---

## 2. Environment setup

We use Python's built-in **`venv`** so nothing beyond a standard Python 3.10
installation is required (no conda needed).

```bash
# 1) clone
git clone https://github.com/YorkXingZeyu/SCV-MPDD.git
cd SCV-MPDD

# 2) create and activate a virtual environment (Python's built-in venv)
python3.10 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

# 3) install PyTorch matching YOUR CUDA version first.
#    Example for CUDA 12.1 (see https://pytorch.org for other CUDA versions):
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 4) install all remaining dependencies
pip install -r requirements.txt

# 5) configure accelerate once (single-GPU default is fine)
accelerate config default
```

### About `requirements.txt`

`requirements.txt` lists the Python packages used by the project and pins the fragile
parts of the stack. In particular, `huggingface_hub==0.19.4` is kept because
`diffusers==0.20.1` still imports the older `cached_download` API. `numpy` is
pinned to the 1.26 series and `opencv-python` to 4.9 to avoid untested NumPy 2.x
or OpenCV 5.x combinations with PyTorch 2.1. Install `torch`/`torchvision` in
step 3 first so pip picks the build matching your CUDA; the same versions remain
in `requirements.txt` as a fallback.

The optional flags in the training script that would require
`bitsandbytes`, `xformers`, or `wandb` are **not used** by the default scripts, so you
do **not** need to install them.

> **Base model.** The scripts download `stabilityai/stable-diffusion-2-inpainting` from
> the Hugging Face Hub on first run (~5 GB). It is cached under `~/.cache/huggingface`.
> To use a custom cache location, export `HF_HOME=/path/to/cache` before running.
> If you are offline **after** the first download, export `HF_HUB_OFFLINE=1`.

Quick sanity check that the environment is OK:

```bash
python -c "import torch, diffusers, transformers, accelerate, peft, cv2, sklearn; \
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
```

---

## 3. Dataset

Download MPDD and place it in **MVTec format** under `data/MPDD`:

```
data/MPDD/
  <category>/
    train/good/*.png                              # normal images (for training)
    test/good/*.png                               # normal test images
    test/<defect_type>/*.png                      # defective test images
    ground_truth/<defect_type>/<name>_mask.png    # pixel masks for defects
```

The six categories are:
`connector, metal_plate, bracket_black, bracket_brown, bracket_white, tubes`.

You can point the scripts at a different location with `--data_root /your/path`.

---

## 4. Quick start (all six categories)

One command runs the whole pipeline — data prep → LoRA fine-tuning → evaluation:

```bash
bash run_all.sh
```

Results are written to `results.csv`:

```
category,image_auroc,pixel_auroc,n_good,n_bad,defects
connector,...
```

Override any default via environment variables, e.g.:

```bash
RES=256 N_RECON=10 STRENGTH=0.3 OUT=results.csv bash run_all.sh
```

---

## 5. Step by step (single category)

Use `connector` as an example; swap in any category name.

### 5.1 Prepare RealFill training data

Builds `runs/connector/data/` (reference images + a target image with a center mask):

```bash
python prep_data.py connector \
    --data_root ./data/MPDD --runs_dir ./runs --resolution 256
```

### 5.2 Fine-tune the per-category LoRA

Trains LoRA adapters on top of SD2-inpainting for that category's normal images.
For evaluation convenience, the script merges the learned adapters back into the
base model and saves a complete Diffusers inpainting pipeline under
`runs/connector/lora/` (the directory name is historical; expect several GB
per category, not a small adapter-only checkpoint):

```bash
RES=256 bash train.sh connector
```

Training defaults (edit in `train.sh` if needed): 1000 steps, batch size 4,
gradient accumulation 2, fp16, LoRA rank 8. Takes roughly 15–25 min on one modern GPU.

### 5.3 Evaluate

Runs the sliding-window reconstruction + AUROC scoring:

```bash
python evaluate.py \
    --data_root ./data/MPDD --runs_dir ./runs \
    --categories connector \
    --resolution 256 --strength 0.3 --n_recon 10 \
    --out results.csv
```

`evaluate.py` is **resumable**: a category already present in `--out` is skipped, so you
can stop and restart without losing progress. Pass several categories comma-separated,
e.g. `--categories connector,tubes`.

### Evaluation options

| flag | default | meaning |
|------|---------|---------|
| `--resolution` | `256` | working resolution `R` |
| `--block` | `R/8` | sliding-window block size `K` |
| `--stride` | `2` | checkerboard stride `S` (`S*S` passes) |
| `--strength` | `0.3` | inpainting strength |
| `--n_recon` | `10` | reconstructions averaged per image |
| `--lora_subdir` | `lora` | subdir under `runs/<cat>/` holding the finetuned inpainting pipeline |

---

## 6. Repository layout

```
SCV-MPDD/
  prep_data.py          # build per-category RealFill training data
  train_realfill.py     # RealFill LoRA fine-tuning (SD2 inpainting)
  train.sh              # training launcher (hyper-parameters)
  evaluate.py           # sliding-window reconstruction + AUROC evaluation
  run_all.sh            # prep -> train -> evaluate over all categories
  requirements.txt
  README.md
  data/                 # place MPDD here (git-ignored)
  runs/                 # prepared data + finetuned inpainting pipelines + results (git-ignored)
```

---

## 7. Troubleshooting

- **`AssertionError: no images found in .../train/good`** — the dataset path is wrong.
  Check `--data_root` and that the MVTec-format folders exist.
- **`missing finetuned model: runs/<cat>/lora`** — you ran `evaluate.py` before training,
  or the finetuned Diffusers pipeline was not saved completely. Run `prep_data.py`
  then `train.sh` for that category first.
- **CUDA out of memory during training** — lower `--train_batch_size` in `train.sh`
  (it already uses gradient checkpointing + fp16), or keep resolution at 256.
- **`accelerate` asks questions / errors on launch** — run `accelerate config default`
  once (single GPU, no distributed, no fp16 prompt needed since `train.sh` sets it).
- **First run hangs at model download** — it is fetching the ~5 GB base model from HF.
  Let it finish once; subsequent runs use the cache.
- **`ImportError: bitsandbytes/xformers/wandb`** — you enabled an optional flag. The
  default scripts don't need these; remove the flag or `pip install` the package.

---

## Acknowledgements

- [RealFill](https://arxiv.org/abs/2309.16668) — reference-based inpainting personalization.
- [Stable Diffusion 2](https://github.com/Stability-AI/stablediffusion) — base inpainting model.
- [MPDD](https://github.com/stepanje/MPDD) — Metal Parts Defect Detection dataset.
