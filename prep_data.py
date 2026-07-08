#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prep_data.py — build RealFill training data for one MPDD category.

RealFill fine-tunes on a small "scene": a set of reference images plus one
target image with a mask. We use all normal ("good") training images of the
category as references, and a center-box mask on one of them as the target.

Produces:
  <runs_dir>/<category>/data/ref/            <- all train/good images, resized
  <runs_dir>/<category>/data/target/target.png
  <runs_dir>/<category>/data/target/mask.png <- center box mask

Usage:
  python prep_data.py connector --data_root ./data/MPDD --runs_dir ./runs --resolution 256
"""
import argparse, glob, os
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("category")
ap.add_argument("--data_root", default="./data/MPDD", help="MPDD root (MVTec-format)")
ap.add_argument("--runs_dir", default="./runs", help="output dir for prepared data / models")
ap.add_argument("--resolution", type=int, default=256)
args = ap.parse_args()

R = args.resolution
src = f"{args.data_root}/{args.category}/train/good"
dst = f"{args.runs_dir}/{args.category}/data"
os.makedirs(f"{dst}/ref", exist_ok=True)
os.makedirs(f"{dst}/target", exist_ok=True)

imgs = sorted(glob.glob(src + "/*"))
assert imgs, f"no images found in {src}"

for f in imgs:
    name = os.path.basename(f).rsplit(".", 1)[0] + ".png"
    Image.open(f).convert("RGB").resize((R, R)).save(f"{dst}/ref/{name}")

# target = first normal image; mask = center box
Image.open(imgs[0]).convert("RGB").resize((R, R)).save(f"{dst}/target/target.png")
m = np.zeros((R, R), np.uint8)
m[R // 4:3 * R // 4, R // 4:3 * R // 4] = 255
Image.fromarray(m).save(f"{dst}/target/mask.png")

print(f"[ok] {args.category}: {len(imgs)} refs @ {R}x{R}, target+mask -> {dst}")
