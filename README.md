# RealFill-MPDD

Anomaly detection on the **MPDD** dataset using per-category **RealFill** fine-tuning
of Stable Diffusion 2 inpainting, with a full-coverage sliding-window
reconstruction-and-compare pipeline.

The core idea: a LoRA-fine-tuned inpainting model, trained **only on normal images**,
learns to reconstruct the *normal* appearance of each object. At test time we mask and
reconstruct the entire image; defective regions get "repaired" to look normal, so the
feature-space difference between the original and the reconstruction lights up exactly
where the defect is.

---

## Method

Given a test image (resized to `R x R`, default `256`):

1. **Full-coverage sliding-window reconstruction.**
   The image is split into a grid of `K x K` blocks (`K = R/8`, i.e. an 8×8 grid).
   A checkerboard schedule with stride `S=2` yields `S*S = 4` passes; each pass masks a
   quarter of non-adjacent blocks so the four passes tile the image **exactly once**.
   Every masked region is inpainted by the category's LoRA model (`strength`, 50 DDPM
   steps, guidance 3.0) and stitched back into a full reconstruction.
   This is repeated `N` times (default `10`) with different seeds and averaged — pure
   denoising that boosts the signal-to-noise ratio for small defects.

2. **Per-layer feature difference.**
   The original and reconstructed images are passed through an ImageNet-pretrained
   **WideResNet-50-2**; per-position squared L2 differences are taken at layers
   `l1`, `l2`, `l3`, upsampled to `R x R`, and Gaussian-blurred (σ=4).

3. **Fixed-scale normalization.**
   Each layer's diff map is divided by a single global scalar — that layer's mean diff
   averaged over the **normal ("good") test images only**. This balances the very
   different magnitudes across layers while preserving the cross-image signal that
   image-level scoring depends on.

4. **Anomaly map.**
   `M = D_l1' + D_l2' + D_l3'`  (all three layers).

5. **Scoring (both metrics from the same map `M`).**
   - **image-AUROC**: image score = mean of the top-1% brightest pixels of `M`.
   - **pixel-AUROC**: `M` compared pixel-wise against the ground-truth mask.

---

## Installation

```bash
git clone <this-repo>
cd RealFill-MPDD
pip install -r requirements.txt
accelerate config default        # or configure interactively
```

The base model `stabilityai/stable-diffusion-2-inpainting` is downloaded from the
Hugging Face Hub on first use (set `HF_HOME` to cache it; use `HF_HUB_OFFLINE=1` for
offline runs once cached).

---

## Data

Place MPDD in **MVTec format** under `data/MPDD`:

```
data/MPDD/
  <category>/
    train/good/*.png                     # normal images (training)
    test/good/*.png                      # normal test images
    test/<defect_type>/*.png             # defective test images
    ground_truth/<defect_type>/<name>_mask.png   # pixel masks
```

Categories: `connector, metal_plate, bracket_black, bracket_brown, bracket_white, tubes`.

---

## Usage

### End-to-end (all six categories)

```bash
bash run_all.sh
# -> results.csv  (category, image_auroc, pixel_auroc, n_good, n_bad, defects)
```

### Step by step (single category)

```bash
# 1. build RealFill training data (references + target/mask)
python prep_data.py connector --data_root ./data/MPDD --runs_dir ./runs --resolution 256

# 2. fine-tune the per-category LoRA
RES=256 bash train.sh connector          # -> runs/connector/lora/

# 3. evaluate
python evaluate.py --data_root ./data/MPDD --runs_dir ./runs \
    --categories connector --resolution 256 --strength 0.3 --n_recon 10 \
    --out results.csv
```

### Key options (`evaluate.py`)

| flag | default | meaning |
|------|---------|---------|
| `--resolution` | `256` | working resolution `R` |
| `--block` | `R/8` | sliding-window block size `K` |
| `--stride` | `2` | checkerboard stride `S` (`S*S` passes) |
| `--strength` | `0.3` | inpainting strength |
| `--n_recon` | `10` | reconstructions averaged per image |

`evaluate.py` is resumable: completed categories are skipped if already present in the
output CSV.

---

## Repository layout

```
RealFill-MPDD/
  prep_data.py          # build per-category RealFill training data
  train_realfill.py     # RealFill LoRA fine-tuning (SD2 inpainting)
  train.sh              # training launcher (hyper-parameters)
  evaluate.py           # sliding-window reconstruction + AUROC evaluation
  run_all.sh            # prep -> train -> evaluate over all categories
  requirements.txt
  data/                 # (place MPDD here; git-ignored)
  runs/                 # prepared data + finetuned models + results (git-ignored)
```

---

## Acknowledgements

- [RealFill](https://arxiv.org/abs/2309.16668) — reference-based inpainting personalization.
- [Stable Diffusion 2](https://github.com/Stability-AI/stablediffusion) — base inpainting model.
- [MPDD](https://github.com/stepanje/MPDD) — Metal Parts Defect Detection dataset.
