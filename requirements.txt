#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — Full evaluation pipeline for NEWMMF paper.

Pipeline:
  1. Load 4 typical color images, convert to grayscale.
  2. For each typical image:
        For each noise density in [50, 60, 70, 80, 90]%:
            Generate noisy image (Bernoulli SPN, seed=42)
            For each method in {DBA, ACWMF, NAFSMF, DAMF, MTA, CANDAR, NEWMMF}:
                Apply method, compute PSNR/SSIM
                Save denoised image to per-method folder
        Save original + noisy versions
  3. Load BSD68 dataset.
     For each image x density x method:
        Compute PSNR/SSIM (DO NOT SAVE denoised images for BSD68).
     Compute average per (method, density).
  4. Save:
        - results_4_typical.csv  (per-image, per-method, per-density)
        - results_BSD68_average.csv  (per-method, per-density average over 68)
        - 10 PNG graphs (4 images x 2 metrics + BSD68 x 2 metrics)

Folder structure produced:
  results/
    Lena/
      original.png
      noisy_50.png ... noisy_90.png
      DBA/    denoised_50.png ... denoised_90.png
      ACWMF/  denoised_50.png ... denoised_90.png
      NAFSMF/ ...
      DAMF/   ...
      MTA/    ...
      CANDAR/ ...
      NEWMMF/   ...
    Barbara/  (same structure)
    Baboon/   (same structure)
    Peppers/  (same structure)
    graphs/   *.png
    csv/      *.csv
"""

import os
import csv
import time
import numpy as np
import cv2
# import os
import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


# Local imports
from utils import (
    add_sp_noise, gray_psnr, gray_ssim, to_uint8,
    setup_output_dirs, save_image, load_grayscale, RANDOM_SEED,
)
from methods_baseline import run_dba, run_acwmf, run_nafsmf, run_damf
from methods_proposed import run_mta, run_candar, run_newmmf
from visualization import plot_all_typical, plot_bsd68_average


# ============================================================================
# Configuration
# ============================================================================
IMAGE_DIR  = "./images"            # 4 typical images (color)
BSD68_DIR  = "./BSD68"             # BSD68 dataset
OUTPUT_DIR = "./results_260703"

# Typical image filenames — adjust extensions to match your local files.
# Set IMAGES = None to auto-detect (any file containing the image keyword).
TYPICAL_IMAGES = {
    "Lena":    None,   # auto-detect
    "Barbara": None,
    "Baboon":  None,
    "Peppers": None,
}

# Noise densities to evaluate (high-density focus)
NOISE_DENSITIES = [0.5, 0.6, 0.7, 0.8, 0.9]

# Method registry: name -> callable(noisy_f64, noise_mask) -> denoised
METHODS = {
    "DBA":    lambda noisy, mask: run_dba(noisy),
    "ACWMF":  lambda noisy, mask: run_acwmf(noisy, s=0.6),
    "NAFSMF": lambda noisy, mask: run_nafsmf(noisy),
    "DAMF":   lambda noisy, mask: run_damf(noisy),
    "MTA":    lambda noisy, mask: run_mta(to_uint8(noisy), max_iterations=1000),
    "CANDAR": lambda noisy, mask: run_candar(noisy),
    "NEWMMF-P1": lambda noisy, mask: run_newmmf(noisy, return_phases=True)[0],
    "NEWMMF":   lambda noisy, mask: run_newmmf(noisy),
}
METHOD_ORDER = ["ACWMF", "DBA", "NAFSMF", "DAMF", "MTA", "CANDAR",
                "NEWMMF-P1", "NEWMMF"]


# ============================================================================
# Image loading helpers
# ============================================================================
def find_typical_image(image_keyword, image_dir):
    """Find a typical image by case-insensitive keyword matching."""
    for fn in sorted(os.listdir(image_dir)):
        name_no_ext = os.path.splitext(fn)[0]
        if image_keyword.lower() in name_no_ext.lower():
            return os.path.join(image_dir, fn)
    return None


def load_typical_images(image_dir, typical_map):
    """
    Load typical images. Returns dict {name: grayscale_uint8}.
    typical_map: {name: filename or None for auto-detect}.
    """
    loaded = {}
    for name, fn in typical_map.items():
        if fn is None:
            path = find_typical_image(name, image_dir)
        else:
            path = os.path.join(image_dir, fn)
        if path is None or not os.path.exists(path):
            print(f"  [WARN] {name} not found in {image_dir} (looked for "
                  f"'{name}' keyword). Skipping.")
            continue
        gray = load_grayscale(path)
        loaded[name] = gray
        print(f"  Loaded {name}: {gray.shape} from {os.path.basename(path)}")
    return loaded


def list_bsd68_images(bsd68_dir):
    """List all valid image files in BSD68 directory."""
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    if not os.path.exists(bsd68_dir):
        return []
    files = sorted([
        os.path.join(bsd68_dir, fn)
        for fn in os.listdir(bsd68_dir)
        if fn.lower().endswith(valid_exts)
    ])
    return files


# ============================================================================
# Evaluation: 4 typical images (per-image folder structure + CSV)
# ============================================================================
def evaluate_typical(images, output_dir):
    """
    Evaluate all methods on all typical images at all densities.
    Saves: original, noisy, per-method denoised images. Returns results dict.

    Returns:
        results_all: {image_name: {"PSNR": {method: [vals]}, "SSIM": {method: [vals]}}}
        rows:        list of CSV rows (Image, Method, Density, PSNR, SSIM)
    """
    setup_output_dirs(output_dir, list(images.keys()), METHOD_ORDER)

    results_all = {}
    rows = []
    rows.append(["Image", "Method", "Density (%)", "PSNR (dB)", "SSIM"])

    for img_name, clean in images.items():
        print(f"\n{'='*70}")
        print(f"  Processing typical image: {img_name}")
        print(f"{'='*70}")

        # Save original
        save_image(clean, os.path.join(output_dir, img_name, "original.png"))

        results_all[img_name] = {
            "PSNR": {m: [] for m in METHOD_ORDER},
            "SSIM": {m: [] for m in METHOD_ORDER},
        }

        for density in NOISE_DENSITIES:
            pct = int(density * 100)
            print(f"\n  --- {img_name} @ {pct}% noise ---")

            # Generate noisy (same seed across methods for fairness)
            noisy_f, noise_mask = add_sp_noise(clean, density, seed=RANDOM_SEED)
            noisy_u8 = to_uint8(noisy_f)

            # Save noisy image
            save_image(
                noisy_u8,
                os.path.join(output_dir, img_name, f"noisy_{pct}.png"),
            )

            # Apply each method
            for method_name in METHOD_ORDER:
                if method_name not in METHODS:
                    continue
                fn = METHODS[method_name]

                t0 = time.time()
                # noisy_f and noise_mask are passed; method may use them
                denoised = fn(noisy_f, noise_mask)
                elapsed = time.time() - t0

                denoised_u8 = to_uint8(denoised)
                p = gray_psnr(clean, denoised_u8)
                s = gray_ssim(clean, denoised_u8)

                results_all[img_name]["PSNR"][method_name].append(p)
                results_all[img_name]["SSIM"][method_name].append(s)
                rows.append([img_name, method_name, pct,
                             f"{p:.4f}", f"{s:.6f}"])

                # Save per-method denoised
                save_image(
                    denoised_u8,
                    os.path.join(
                        output_dir, img_name, method_name,
                        f"denoised_{pct}.png",
                    ),
                )

                print(f"    {method_name:<7s}: PSNR={p:6.2f} dB, "
                      f"SSIM={s:.4f}, time={elapsed:6.2f}s")

    return results_all, rows


# ============================================================================
# Evaluation: BSD68 (average only, NO image saving)
# ============================================================================
def evaluate_bsd68(bsd68_files):
    """
    Evaluate all methods on all BSD68 images at all densities.
    Returns only AVERAGE PSNR/SSIM (no per-image storage).

    Returns:
        results_bsd68: {"PSNR": {method: [avg_per_density]},
                        "SSIM": {method: [avg_per_density]}}
        rows:          list of CSV rows for BSD68 average
    """
    n_images = len(bsd68_files)
    print(f"\n{'='*70}")
    print(f"  BSD68 evaluation: {n_images} images")
    print(f"{'='*70}")

    # Accumulators: sums per (metric, method, density)
    psnr_sum = {m: [0.0] * len(NOISE_DENSITIES) for m in METHOD_ORDER}
    ssim_sum = {m: [0.0] * len(NOISE_DENSITIES) for m in METHOD_ORDER}
    count = [0] * len(NOISE_DENSITIES)  # actually same across densities, but track per density

    for img_idx, img_path in enumerate(bsd68_files, start=1):
        try:
            clean = load_grayscale(img_path)
        except Exception as e:
            print(f"  [WARN] Failed to load {img_path}: {e}")
            continue

        print(f"\n  [{img_idx}/{n_images}] {os.path.basename(img_path)} "
              f"({clean.shape})")

        for d_idx, density in enumerate(NOISE_DENSITIES):
            pct = int(density * 100)
            noisy_f, noise_mask = add_sp_noise(clean, density, seed=RANDOM_SEED)

            for method_name in METHOD_ORDER:
                if method_name not in METHODS:
                    continue
                fn = METHODS[method_name]
                denoised = fn(noisy_f, noise_mask)
                d_u8 = to_uint8(denoised)
                p = gray_psnr(clean, d_u8)
                s = gray_ssim(clean, d_u8)
                psnr_sum[method_name][d_idx] += p
                ssim_sum[method_name][d_idx] += s

            count[d_idx] += 1

        # Quick progress every 10 images
        if img_idx % 10 == 0 or img_idx == n_images:
            print(f"     ... progress {img_idx}/{n_images} done")

    # Compute averages
    results_bsd68 = {
        "PSNR": {m: [psnr_sum[m][i] / count[i] if count[i] > 0 else 0.0
                     for i in range(len(NOISE_DENSITIES))]
                 for m in METHOD_ORDER},
        "SSIM": {m: [ssim_sum[m][i] / count[i] if count[i] > 0 else 0.0
                     for i in range(len(NOISE_DENSITIES))]
                 for m in METHOD_ORDER},
    }

    # Build CSV rows
    rows = [["Method", "Density (%)", "Avg PSNR (dB)", "Avg SSIM",
             "N (images)"]]
    for method_name in METHOD_ORDER:
        for i, density in enumerate(NOISE_DENSITIES):
            pct = int(density * 100)
            rows.append([
                method_name, pct,
                f"{results_bsd68['PSNR'][method_name][i]:.4f}",
                f"{results_bsd68['SSIM'][method_name][i]:.6f}",
                count[i],
            ])

    return results_bsd68, rows


# ============================================================================
# CSV writers
# ============================================================================
def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    print(f"  Saved CSV: {path}")


# ============================================================================
# Main
# ============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "csv"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "graphs"), exist_ok=True)

    print("=" * 70)
    print("NEWMMF Paper — Full Evaluation Pipeline")
    print("=" * 70)
    print(f"  Image dir:        {IMAGE_DIR}")
    print(f"  BSD68 dir:        {BSD68_DIR}")
    print(f"  Output dir:       {OUTPUT_DIR}")
    print(f"  Noise densities:  {[int(d*100) for d in NOISE_DENSITIES]}%")
    print(f"  Methods ({len(METHOD_ORDER)}):      {METHOD_ORDER}")
    print(f"  Noise model:      per-pixel Bernoulli (seed={RANDOM_SEED})")
    print("=" * 70)

    # ------------------------------------------------------------------ Phase A
    # 4 typical images
    print("\nPhase A: Loading 4 typical images")
    images = load_typical_images(IMAGE_DIR, TYPICAL_IMAGES)
    if not images:
        print("[ERROR] No typical images loaded. Check IMAGE_DIR and filenames.")
        print(f"        Expected directory: {IMAGE_DIR}")
        return

    print(f"\nPhase B: Evaluating {len(images)} typical images...")
    t_phase_start = time.time()
    results_typical, rows_typical = evaluate_typical(images, OUTPUT_DIR)
    print(f"\n  Phase B done in {time.time() - t_phase_start:.1f}s")

    # Save CSV
    write_csv(os.path.join(OUTPUT_DIR, "csv",
                           "results_4_typical.csv"), rows_typical)

    # Plot 4 typical x 2 metrics = 8 graphs
    print("\nPhase C: Generating typical-image graphs (8 PNGs)")
    plot_all_typical(NOISE_DENSITIES, results_typical,
                     os.path.join(OUTPUT_DIR, "graphs"),
                     image_names=list(images.keys()))
    print("  Graphs saved.")

    # ------------------------------------------------------------------ Phase D
    # BSD68
    print("\nPhase D: Loading BSD68")
    bsd68_files = list_bsd68_images(BSD68_DIR)
    print(f"  Found {len(bsd68_files)} BSD68 images")

    if bsd68_files:
        print(f"\nPhase E: Evaluating BSD68...")
        t_phase_start = time.time()
        results_bsd68, rows_bsd68 = evaluate_bsd68(bsd68_files)
        print(f"\n  Phase E done in {time.time() - t_phase_start:.1f}s")

        write_csv(os.path.join(OUTPUT_DIR, "csv",
                               "results_BSD68_average.csv"), rows_bsd68)

        print("\nPhase F: Generating BSD68 graphs (2 PNGs)")
        plot_bsd68_average(NOISE_DENSITIES, results_bsd68,
                           os.path.join(OUTPUT_DIR, "graphs"))
        print("  Graphs saved.")
    else:
        print("  [WARN] BSD68 directory empty or missing. Skipping BSD68.")

    # ------------------------------------------------------------------ Summary
    print("\n" + "=" * 70)
    print("ALL DONE")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"    - {len(images)} image folders (original + noisy + 7 methods)")
    print(f"    - graphs/  (10 PNGs)")
    print(f"    - csv/     (2 CSV files)")
    print("=" * 70)


if __name__ == "__main__":
    main()