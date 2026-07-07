#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualization.py — Comparison plots for SPN denoising evaluation.

Produces 10 graphs total:
  - Per typical image (Lena, Barbara, Baboon, Peppers): PSNR + SSIM (4 x 2 = 8)
  - BSD68 dataset average: PSNR + SSIM (2)

Design:
  - x-axis: noise density (%)
  - y-axis: metric (PSNR in dB, or SSIM)
  - Method-coded colors with the proposed method (NEWMMF) highlighted in
    bold red with star marker.
  - Color-blind friendly palette for the 6 baselines.
"""

import os
import matplotlib
matplotlib.use('Agg')  # headless safe
import matplotlib.pyplot as plt


# Method order and colors (color-blind friendly; NEWMMF emphasized)
METHOD_STYLE = {
    "DBA":    {"color": "#0072B2", "marker": "o", "linewidth": 1.6, "linestyle": "--"},
    "ACWMF":  {"color": "#009E73", "marker": "s", "linewidth": 1.6, "linestyle": "--"},
    "NAFSMF": {"color": "#F0E442", "marker": "^", "linewidth": 1.6, "linestyle": "--"},
    "DAMF":   {"color": "#56B4E9", "marker": "D", "linewidth": 1.6, "linestyle": "--"},
    "MTA":    {"color": "#CC79A7", "marker": "v", "linewidth": 1.6, "linestyle": "-."},
    "CANDAR": {"color": "#E69F00", "marker": "P", "linewidth": 1.6, "linestyle": "-."},
    # Proposed method: bold, red, larger
    "NEWMMF":   {"color": "#D55E00", "marker": "*", "linewidth": 2.8,
               "linestyle": "-", "markersize": 14},
}

# Order to plot (last = NEWMMF on top)
METHOD_ORDER = ["DBA", "ACWMF", "NAFSMF", "DAMF", "MTA", "CANDAR", "NEWMMF"]


def plot_single_image_metric(
    densities, results, image_name, metric_name, save_path,
    methods=None,
):
    """
    Plot one metric (PSNR or SSIM) vs noise density for one image.

    Args:
        densities:   list of density values (0-1, e.g. [0.5, 0.6, 0.7, 0.8, 0.9])
        results:     dict of {method_name: list of metric values (one per density)}
        image_name:  string for plot title (e.g. "Lena")
        metric_name: "PSNR" or "SSIM"
        save_path:   output PNG path
        methods:     optional list of method names to plot (defaults to METHOD_ORDER
                     intersected with available results)
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    if methods is None:
        methods = [m for m in METHOD_ORDER if m in results]

    pcts = [int(d * 100) for d in densities]

    for method in methods:
        if method not in results:
            continue
        style = METHOD_STYLE.get(method, {})
        markersize = style.get("markersize", 8)
        ax.plot(
            pcts, results[method],
            color=style.get("color", "gray"),
            marker=style.get("marker", "o"),
            linewidth=style.get("linewidth", 1.6),
            linestyle=style.get("linestyle", "-"),
            markersize=markersize,
            label=method,
        )

    ax.set_xlabel("Noise Density (%)", fontsize=12)
    ax.set_ylabel(f"{metric_name}" + (" (dB)" if metric_name == "PSNR" else ""), fontsize=12)
    ax.set_title(f"{image_name} — {metric_name} vs Noise Density",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(pcts)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all_typical(densities, results_all, output_dir, image_names=None):
    """
    Generate all 8 plots for 4 typical images (4 x PSNR + 4 x SSIM).

    Args:
        densities:    list of density values
        results_all:  nested dict {image_name: {metric: {method: [values]}}}
                      e.g. results_all["Lena"]["PSNR"]["NEWMMF"] = [29.96, ...]
        output_dir:   directory to save PNGs (will be created if missing)
        image_names:  optional ordered list of image names
    """
    os.makedirs(output_dir, exist_ok=True)

    if image_names is None:
        image_names = list(results_all.keys())

    for img_name in image_names:
        if img_name not in results_all:
            continue
        for metric in ["PSNR", "SSIM"]:
            if metric not in results_all[img_name]:
                continue
            save_path = os.path.join(output_dir, f"{img_name}_{metric}.png")
            plot_single_image_metric(
                densities, results_all[img_name][metric],
                image_name=img_name,
                metric_name=metric,
                save_path=save_path,
            )


def plot_bsd68_average(densities, results_bsd68, output_dir):
    """
    Generate 2 plots for BSD68 average (PSNR + SSIM).

    Args:
        densities:     list of density values
        results_bsd68: dict {metric: {method: [average values]}}
        output_dir:    directory to save PNGs
    """
    os.makedirs(output_dir, exist_ok=True)
    for metric in ["PSNR", "SSIM"]:
        if metric not in results_bsd68:
            continue
        save_path = os.path.join(output_dir, f"BSD68_{metric}.png")
        plot_single_image_metric(
            densities, results_bsd68[metric],
            image_name="BSD68 (Average over 68 images)",
            metric_name=metric,
            save_path=save_path,
        )
