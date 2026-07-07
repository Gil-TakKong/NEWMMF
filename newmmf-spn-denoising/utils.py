#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py — Common utilities for SPN denoising evaluation

Functions:
  - add_sp_noise:   per-pixel Bernoulli SPN (matches NEWMMF reference)
  - gray_psnr:      PSNR with data_range=255
  - gray_ssim:      SSIM with data_range=255
  - to_uint8:       clip + cast for display/saving
  - setup_output_dirs: folder structure builder
  - save_image:     PNG save
"""

import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as _skim_ssim

RANDOM_SEED = 42


# ============================================================
# Noise generation (matches NEWMMF reference implementation)
# ============================================================
def add_sp_noise(img, ratio, seed=RANDOM_SEED):
    """
    Add salt-and-pepper noise per-pixel (Bernoulli model).

    Identical to NEWMMF reference (260519_final_method_gray_.py) for
    fair comparison across all methods.

    Args:
        img:   2D grayscale image (uint8 or convertible)
        ratio: noise density in [0.0, 1.0]
        seed:  random seed (default RANDOM_SEED=42)

    Returns:
        noisy: float64 image, corrupted pixels set to 0 or 255
        mask:  boolean array, True where pixel is corrupted
    """
    rng = np.random.default_rng(seed)
    noisy = img.copy().astype(np.float64)
    h, w = img.shape
    mask = rng.random((h, w)) < ratio
    coin = rng.random((h, w))
    noisy[mask & (coin < 0.5)] = 0.0
    noisy[mask & (coin >= 0.5)] = 255.0
    return noisy, mask


# ============================================================
# Metrics
# ============================================================
def gray_psnr(clean, restored):
    """PSNR (dB) for 8-bit grayscale images. data_range=255."""
    c = clean.astype(np.float64)
    r = restored.astype(np.float64)
    mse = np.mean((c - r) ** 2)
    if mse == 0.0:
        return 100.0
    return 10.0 * np.log10(255.0 ** 2 / mse)


def gray_ssim(clean, restored):
    """SSIM for 8-bit grayscale images. data_range=255."""
    return _skim_ssim(
        clean.astype(np.uint8),
        np.clip(restored, 0, 255).astype(np.uint8),
        data_range=255,
    )


# ============================================================
# I/O helpers
# ============================================================
def to_uint8(img):
    """Clip to [0, 255] and cast to uint8 (for display/saving)."""
    return np.clip(img, 0, 255).astype(np.uint8)


def ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_image(img, path):
    """Save grayscale image as PNG (8-bit)."""
    cv2.imwrite(path, to_uint8(img))


def setup_output_dirs(base_dir, image_names, method_names):
    """
    Create folder structure for 4 typical images:

        base_dir/
          <ImageName>/
            <MethodName>/
          graphs/
          csv/
    """
    ensure_dir(base_dir)
    ensure_dir(os.path.join(base_dir, "graphs"))
    ensure_dir(os.path.join(base_dir, "csv"))
    for img in image_names:
        ensure_dir(os.path.join(base_dir, img))
        for method in method_names:
            ensure_dir(os.path.join(base_dir, img, method))


def load_grayscale(path):
    """Load image and convert to grayscale uint8."""
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
