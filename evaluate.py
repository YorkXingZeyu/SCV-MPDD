#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py — RealFill-based anomaly detection on MPDD.

For each test image:
  1. Full-coverage sliding-window reconstruction with the category's LoRA-finetuned
     SD2-inpainting model (checkerboard K-blocks, S x S = 4 passes tiling the whole
     image exactly once), repeated N times with different seeds and averaged.
  2. Per-layer feature difference (WideResNet-50-2, layers l1/l2/l3) between the
     original and reconstructed image, upsampled to full resolution + Gaussian blur.
  3. Fixed-scale normalization: each layer's diff map is divided by that layer's mean
     diff averaged over the "good" (normal) test images only -- a single global scalar
     per (category, layer). This balances layer magnitudes while preserving the
     cross-image signal that image-level scoring relies on.
  4. Anomaly map  M = D_l1' + D_l2' + D_l3'   (layer set "B").
  5. image-AUROC: mean of the top-1% brightest pixels of M, per image.
     pixel-AUROC: M compared pixel-wise against the ground-truth mask.

Both metrics are computed from the SAME map M (consistent evaluation).

Usage:
  python evaluate.py --data_root ./data/MPDD --runs_dir ./runs \
      --categories connector,metal_plate,... --resolution 256 \
      --strength 0.3 --n_recon 10 --out results.csv
"""
import warnings, logging, glob, os, argparse
warnings.filterwarnings("ignore")
for _n in ("transformers", "huggingface_hub", "diffusers"):
    logging.getLogger(_n).setLevel(logging.ERROR)
import numpy as np, cv2, torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel
import torchvision.models as models
from torchvision.transforms import Normalize
from sklearn.metrics import roc_auc_score

ALL_CATEGORIES = ["connector", "metal_plate", "bracket_black",
                  "bracket_brown", "bracket_white", "tubes"]
BASE_MODEL = "stabilityai/stable-diffusion-2-inpainting"
GUIDANCE = 3.0
NUM_INFERENCE_STEPS = 50
PROMPT = "a photo of sks"


def build_feature_extractor(device):
    wr = models.wide_resnet50_2(weights="IMAGENET1K_V1").eval().to(device)
    feats = {}
    wr.layer1.register_forward_hook(lambda a, b, o: feats.__setitem__("l1", o.detach()))
    wr.layer2.register_forward_hook(lambda a, b, o: feats.__setitem__("l2", o.detach()))
    wr.layer3.register_forward_hook(lambda a, b, o: feats.__setitem__("l3", o.detach()))
    norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def ext(im):  # im: HxWx3 uint8
        x = norm(torch.from_numpy(im).permute(2, 0, 1).float() / 255).unsqueeze(0).to(device)
        with torch.no_grad():
            wr(x)
        return {k: v.clone() for k, v in feats.items()}
    return ext


def load_pipe(lora_dir, device):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32,
        safety_checker=None, requires_safety_checker=False)
    pipe.unet = UNet2DConditionModel.from_pretrained(lora_dir, subfolder="unet")
    pipe.text_encoder = CLIPTextModel.from_pretrained(lora_dir, subfolder="text_encoder")
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    return pipe.to(device)


def layer_diffs(pipe, ext, op, orig, strength, R, K, S, n_recon, device):
    """Full-coverage reconstruction averaged over n_recon runs.
       Returns dict of raw (unnormalized) per-layer diff maps."""
    acc = {k: np.zeros((R, R), np.float32) for k in ("l1", "l2", "l3")}
    for s in range(n_recon):
        rec = orig.astype(np.float32).copy()
        for ri in range(S):
            for ci in range(S):
                mm = np.zeros((R, R), np.uint8)
                for r in range(R // K):
                    for c in range(R // K):
                        if r % S == ri and c % S == ci:
                            mm[r*K:r*K+K, c*K:c*K+K] = 255
                gg = torch.Generator(device=device).manual_seed(42 + s)
                out = pipe(prompt=PROMPT, image=op, mask_image=Image.fromarray(mm),
                           num_inference_steps=NUM_INFERENCE_STEPS, strength=strength,
                           guidance_scale=GUIDANCE, generator=gg).images[0].resize((R, R))
                out = np.array(out).astype(np.float32)
                m = mm > 0
                rec[m] = out[m]
        fo, fr = ext(orig), ext(rec.astype(np.uint8))
        for k in ("l1", "l2", "l3"):
            d = ((fo[k] - fr[k]) ** 2).sum(1, keepdim=True).sqrt()
            dd = torch.nn.functional.interpolate(d, size=(R, R), mode="bilinear",
                                                 align_corners=False)[0, 0].cpu().numpy()
            acc[k] += cv2.GaussianBlur(dd, (0, 0), 4)
    return {k: v / n_recon for k, v in acc.items()}


def image_score(a):
    """Image-level score = mean of the top-1% brightest pixels."""
    return float(np.mean(np.sort(a.ravel())[-max(1, int(a.size * 0.01)):]))


def load_gt(data_root, cat, sub, name, R):
    if sub == "good":
        return np.zeros((R, R), bool)
    p = f"{data_root}/{cat}/ground_truth/{sub}/{name}_mask.png"
    if not os.path.exists(p):
        c = glob.glob(f"{data_root}/{cat}/ground_truth/{sub}/{name}*")
        p = c[0] if c else None
    if p is None:
        return np.zeros((R, R), bool)
    return np.array(Image.open(p).convert("L").resize((R, R))) > 127


def pixel_auroc(gts, maps):
    py = np.concatenate([g.ravel() for g in gts]).astype(np.uint8)
    ps = np.concatenate([m.ravel() for m in maps])
    if py.size > 4_000_000:
        idx = np.random.RandomState(0).choice(py.size, 4_000_000, replace=False)
        py, ps = py[idx], ps[idx]
    return roc_auc_score(py, ps) * 100 if py.sum() > 0 else float("nan")


def eval_category(pipe, ext, data_root, cat, R, K, S, strength, n_recon, device):
    test_dir = f"{data_root}/{cat}/test"
    subs = [d for d in sorted(os.listdir(test_dir)) if os.path.isdir(f"{test_dir}/{d}")]
    defects = "|".join(d for d in subs if d != "good")
    ops, origs, labels, gts = [], [], [], []
    for sub in subs:
        for f in sorted(glob.glob(f"{test_dir}/{sub}/*")):
            nm = os.path.basename(f).rsplit(".", 1)[0]
            op = Image.open(f).convert("RGB").resize((R, R))
            ops.append(op); origs.append(np.array(op))
            labels.append(0 if sub == "good" else 1)
            gts.append(load_gt(data_root, cat, sub, nm, R))
    good = [i for i, l in enumerate(labels) if l == 0]

    diffs = [layer_diffs(pipe, ext, ops[i], origs[i], strength, R, K, S, n_recon, device)
             for i in range(len(ops))]
    # fixed-scale normalization: per-layer scalar = mean over good images
    scale = {k: (np.mean([diffs[i][k].mean() for i in good]) + 1e-8)
             for k in ("l1", "l2", "l3")}
    # anomaly map M = l1' + l2' + l3'  (layer set "B")
    maps = [sum(diffs[i][k] / scale[k] for k in ("l1", "l2", "l3"))
            for i in range(len(ops))]

    img_auroc = roc_auc_score(labels, [image_score(m) for m in maps]) * 100
    pix_auroc = pixel_auroc(gts, maps)
    return img_auroc, pix_auroc, labels.count(0), labels.count(1), defects


def main():
    ap = argparse.ArgumentParser(description="RealFill anomaly detection on MPDD (layer set B)")
    ap.add_argument("--data_root", default="./data/MPDD", help="MPDD root (MVTec-format)")
    ap.add_argument("--runs_dir", default="./runs", help="dir holding <cat>/lora finetuned models")
    ap.add_argument("--categories", default=",".join(ALL_CATEGORIES),
                    help="comma-separated categories")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--block", type=int, default=None,
                    help="sliding-window block size K (default resolution/8 -> 8x8 grid)")
    ap.add_argument("--stride", type=int, default=2, help="checkerboard stride S (SxS passes)")
    ap.add_argument("--strength", type=float, default=0.3)
    ap.add_argument("--n_recon", type=int, default=10, help="reconstructions averaged per image")
    ap.add_argument("--lora_subdir", default="lora")
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    R = args.resolution
    K = args.block if args.block else R // 8
    S = args.stride
    cats = [c for c in args.categories.split(",") if c in ALL_CATEGORIES]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[cfg] R={R} K={K} S={S} strength={args.strength} n_recon={args.n_recon} "
          f"cats={cats}", flush=True)

    ext = build_feature_extractor(device)

    # resumable CSV
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            p = line.strip().split(",")
            if len(p) >= 1 and p[0] != "category":
                done.add(p[0])
    else:
        with open(args.out, "w") as f:
            f.write("category,image_auroc,pixel_auroc,n_good,n_bad,defects\n")

    for cat in cats:
        if cat in done:
            print(f"[skip] {cat}", flush=True); continue
        lora_dir = f"{args.runs_dir}/{cat}/{args.lora_subdir}"
        assert os.path.exists(f"{lora_dir}/model_index.json"), \
            f"missing finetuned model: {lora_dir} (run training first)"
        pipe = load_pipe(lora_dir, device)
        img, pix, ng, nb, defects = eval_category(
            pipe, ext, args.data_root, cat, R, K, S, args.strength, args.n_recon, device)
        with open(args.out, "a") as f:
            f.write(f"{cat},{img:.2f},{pix:.2f},{ng},{nb},{defects}\n")
        print(f"[ok] {cat}: image-AUROC={img:.2f} pixel-AUROC={pix:.2f}", flush=True)
        del pipe; torch.cuda.empty_cache()
    print("===== done =====", flush=True)


if __name__ == "__main__":
    main()
