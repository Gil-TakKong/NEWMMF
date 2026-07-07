#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
methods_proposed.py — Self-citation and proposed methods.

Implementations in this file:
  - MTA    (Kong & Choi, MTA 2026, self-citation)
           Verbatim re-implementation of user's 251207_Fixed_Iter_histogram.py
           (per user instruction: preserve user's algorithm exactly).

  - CANDAR (Kong & Choi, CANDAR 2025 conference, self-citation)
           Verbatim re-implementation of user's _2025CANDAR_SPN_variation.py
           Two-phase: ortho/diagonal directional candidates + variance-weighted
           fusion.

  - NEWMMF   (Kong & Choi, this paper - PROPOSED)
           Wrapped directly from user's reference implementation
           (260519_final_method_gray_.py). Uses numba JIT for execution speed
           while keeping the exact algorithm and parameters.

Public API:
  - run_mta(noisy, max_iterations=1000)
  - run_candar(noisy, max_iter=30, window_size=3)
  - run_newmmf(noisy, noise_mask=None)
"""

import numpy as np
from numba import njit
from helpers import reflect_index, is_clean, median_array


# ============================================================================
# MTA — Modified Two-phase Adaptive (Kong & Choi, 2026, MTA self-cite)
#
# Algorithm (verbatim from user's _-_250218_Denoising_save.py):
#   1. Von Neumann 4 groups, each with 4 directions (Center C included):
#         Group 1: N, W, C, E
#         Group 2: W, N, C, S
#         Group 3: W, C, E, S
#         Group 4: S, C, N, E
#   2. For each noisy pixel (value 0 or 255):
#        For each group (in order):
#          - Collect group values (None if out-of-bounds).
#          - Filter out {0, 255, None}.
#          - If at least 2 non-noise values remain: replace with their mean,
#            then break (do not try later groups).
#        If no group satisfies the condition: leave pixel unchanged.
#   3. Iterate until no 0/255 remain or max_iterations is reached
#      (paper default 20).
# ============================================================================

# Von Neumann neighbor offsets
_NEIGHBOR_OFFSETS = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1)}

# 4 groups of 4 directions each (Center "C" included)
_NEIGHBOR_GROUPS = {
    "Group 1": ["N", "W", "C", "E"],
    "Group 2": ["W", "N", "C", "S"],
    "Group 3": ["W", "C", "E", "S"],
    "Group 4": ["S", "C", "N", "E"],
}


def _get_group_values(grid, i, j, group_name):
    """
    Get values of neighbors in a group; None for out-of-bounds.
    Direct port of user's get_group_values function.
    """
    group = _NEIGHBOR_GROUPS[group_name]
    group_values = []
    for direction in group:
        if direction == "C":
            group_values.append(grid[i, j])
        else:
            di, dj = _NEIGHBOR_OFFSETS[direction]
            ni, nj = i + di, j + dj
            if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                group_values.append(grid[ni, nj])
            else:
                group_values.append(None)
    return group_values


def _mta_one_pass(grid):
    """
    Single pass of MTA: replace each noisy pixel with mean of first group
    containing >= 2 non-noise values. Direct port of user's denoise_image.
    """
    rows, cols = grid.shape
    denoised_grid = grid.copy()
    for i in range(rows):
        for j in range(cols):
            if denoised_grid[i, j] in {0, 255}:
                noise_removed = False
                for group_name in _NEIGHBOR_GROUPS.keys():
                    group_values = _get_group_values(grid, i, j, group_name)
                    non_noise_values = [
                        val for val in group_values
                        if val not in {0, 255, None}
                    ]
                    if len(non_noise_values) >= 2:
                        denoised_grid[i, j] = int(np.mean(non_noise_values))
                        noise_removed = True
                        break
                if not noise_removed:
                    denoised_grid[i, j] = grid[i, j]
    return denoised_grid


def run_mta(noisy, max_iterations=1000):
    """
    Public API for MTA.

    Args:
        noisy:          2D numpy array (uint8), 0/255 SPN corruption.
        max_iterations: maximum passes (paper default 1000; from
                        251207_Fixed_Iter_histogram.py).

    Returns:
        Denoised image as uint8.

    Notes:
        - Identical to user's 251207_Fixed_Iter_histogram.py
        - max_iterations=1000 is critical for high-density (90%) noise:
          at 90% density, ~65% of windows have all 4-group positions
          corrupted, so many passes are needed for noise to propagate.
        - Safety: if noise_count is unchanged for 10 consecutive iterations,
          break (prevents infinite loop in pathological cases).
    """
    denoised_grid = np.clip(noisy, 0, 255).astype(np.uint8).copy()
    iteration_count = 0
    noise_history = []

    while iteration_count < max_iterations:
        noise_count = int(np.sum(
            (denoised_grid == 0) | (denoised_grid == 255)
        ))
        if noise_count == 0:
            break

        # Safety: 10 consecutive identical noise_count -> break
        noise_history.append(noise_count)
        if len(noise_history) > 10:
            recent = noise_history[-10:]
            if len(set(recent)) == 1 and recent[0] > 0:
                break

        denoised_grid = _mta_one_pass(denoised_grid)
        iteration_count += 1

    return denoised_grid


# ============================================================================
# CANDAR — Two-phase Directional Candidate + Variance-Weighted Fusion
# Reference:
#   Kong & Choi, CANDAR 2025 (conference, self-citation).
#   Verbatim port of user's _2025CANDAR_SPN_variation.py.
#
# Algorithm:
#   Two configurations:
#     - Ortho: 4 cardinal direction groups, each with 3 cardinal neighbors.
#         N group:  (-1, 0), ( 0,-1), ( 0, 1)
#         S group:  ( 1, 0), ( 0,-1), ( 0, 1)
#         W group:  (-1, 0), ( 1, 0), ( 0,-1)
#         E group:  (-1, 0), ( 1, 0), ( 0, 1)
#     - Diagonal: 4 diagonal direction groups, each with 3 diagonal neighbors.
#         N diag:  (-1,-1), ( 1,-1), (-1, 1)
#         S diag:  ( 1, 1), ( 1,-1), (-1, 1)
#         W diag:  (-1,-1), ( 1,-1), ( 1, 1)
#         E diag:  (-1,-1), (-1, 1), ( 1, 1)
#
#   For each noisy pixel:
#     1. Compute ortho candidate:
#         - For each direction group: count valid (non 0/255) neighbors.
#         - If valid > total/2 (i.e., >= 2): candidate = mean of valid pixels.
#         - Pick group with highest valid count.
#         - If tied: pick group with minimum variance of valid pixels.
#     2. Compute diagonal candidate (same procedure).
#     3. Fusion:
#         - Both valid: variance-weighted mean (lower variance = higher weight)
#         - One valid: use that one
#         - Neither: no update
#   Iterate until no noise or max_iter=30.
# ============================================================================

def _candar_get_configs():
    """Direction configurations (ortho + diagonal)."""
    configs = {}
    ortho_dirs = {
        "N": [(-1, 0), (0, -1), (0, 1)],
        "S": [(1, 0), (0, -1), (0, 1)],
        "W": [(-1, 0), (1, 0), (0, -1)],
        "E": [(-1, 0), (1, 0), (0, 1)],
    }
    configs['ortho'] = {'dirs': ortho_dirs}
    diag_dirs = {
        "N": [(-1, -1), (1, -1), (-1, 1)],
        "S": [(1, 1), (1, -1), (-1, 1)],
        "W": [(-1, -1), (1, -1), (1, 1)],
        "E": [(-1, -1), (-1, 1), (1, 1)],
    }
    configs['diagonal'] = {'dirs': diag_dirs}
    return configs


def _candar_calculate_candidate(img, x, y, h, w, config):
    """
    Compute directional candidate value with variance tie-breaking.
    Direct port of user's calculate_candidate_value.
    Returns -1.0 if no valid group exists.
    """
    best_score = -1.0
    tied_candidates = []

    for dkey in ("N", "W", "S", "E"):
        offs = config['dirs'][dkey]
        vals_list = []
        for dx, dy in offs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w:
                v = img[nx, ny]
                if v != 0 and v != 255:
                    vals_list.append(v)

        valid_w = len(vals_list)
        total_w = len(offs)

        # Only process if valid count > half (i.e., >= 2 out of 3)
        if valid_w > total_w / 2:
            cand = float(np.mean(np.array(vals_list)))

            if valid_w > best_score:
                best_score = valid_w
                tied_candidates = [(cand, vals_list)]
            elif valid_w == best_score:
                tied_candidates.append((cand, vals_list))

    if not tied_candidates:
        return -1.0

    if len(tied_candidates) == 1:
        return tied_candidates[0][0]

    # Tie-break: pick group with minimum variance
    min_variance = float('inf')
    best_val_on_tie = -1.0
    for cand_val, pixel_vals in tied_candidates:
        variance = (
            float(np.var(np.array(pixel_vals)))
            if len(pixel_vals) > 1 else 0.0
        )
        if variance < min_variance:
            min_variance = variance
            best_val_on_tie = cand_val
    return best_val_on_tie


@njit(cache=True)
def _candar_fuse_jit(img, img_next, noise_locations,
                     cand_ortho_arr, cand_diag_arr, pad_size):
    """
    JIT-compiled fusion + update.
    Variance-weighted fusion of ortho and diagonal candidates.
    Direct port of user's _fuse_and_update_jit.
    """
    h, w = img.shape
    changed_count = 0

    for i in range(noise_locations.shape[0]):
        x = noise_locations[i, 0]
        y = noise_locations[i, 1]

        cand_ortho = cand_ortho_arr[x, y]
        cand_diag = cand_diag_arr[x, y]
        final_val = -1.0

        if cand_ortho != -1.0 and cand_diag != -1.0:
            var_ortho = -1.0
            var_diag = -1.0

            # Ortho variance from window of clean pixels (substitute center with cand_ortho)
            n_o = 0
            sum_o = 0.0
            sum_sq_o = 0.0
            for r_off in range(-pad_size, pad_size + 1):
                for c_off in range(-pad_size, pad_size + 1):
                    px = x + r_off
                    py = y + c_off
                    if 0 <= px < h and 0 <= py < w:
                        if r_off == 0 and c_off == 0:
                            val = cand_ortho
                        else:
                            val = img[px, py]
                        if val != 0.0 and val != 255.0:
                            n_o += 1
                            sum_o += val
                            sum_sq_o += val * val
            if n_o > 1:
                mean_o = sum_o / n_o
                var_ortho = sum_sq_o / n_o - mean_o * mean_o

            # Diag variance (substitute center with cand_diag)
            n_d = 0
            sum_d = 0.0
            sum_sq_d = 0.0
            for r_off in range(-pad_size, pad_size + 1):
                for c_off in range(-pad_size, pad_size + 1):
                    px = x + r_off
                    py = y + c_off
                    if 0 <= px < h and 0 <= py < w:
                        if r_off == 0 and c_off == 0:
                            val = cand_diag
                        else:
                            val = img[px, py]
                        if val != 0.0 and val != 255.0:
                            n_d += 1
                            sum_d += val
                            sum_sq_d += val * val
            if n_d > 1:
                mean_d = sum_d / n_d
                var_diag = sum_sq_d / n_d - mean_d * mean_d

            # Variance-weighted fusion
            is_o_inf = (var_ortho == -1.0)
            is_d_inf = (var_diag == -1.0)
            if is_o_inf and is_d_inf:
                final_val = (cand_ortho + cand_diag) / 2.0
            elif is_o_inf:
                final_val = cand_diag
            elif is_d_inf:
                final_val = cand_ortho
            else:
                eps = 1e-10
                inv_o = 1.0 / (var_ortho + eps)
                inv_d = 1.0 / (var_diag + eps)
                w_o = inv_o / (inv_o + inv_d)
                final_val = w_o * cand_ortho + (1.0 - w_o) * cand_diag

        elif cand_ortho != -1.0:
            final_val = cand_ortho
        elif cand_diag != -1.0:
            final_val = cand_diag

        if final_val != -1.0:
            img_next[x, y] = final_val
            changed_count += 1

    return changed_count


def run_candar(noisy, max_iter=30, window_size=3):
    """
    Public API for CANDAR.

    Args:
        noisy:       2D numpy array, SPN-corrupted image.
        max_iter:    maximum iterations (paper default 30).
        window_size: window for variance calculation (paper default 3 -> 3x3).

    Returns:
        Denoised image as uint8.

    Notes:
        - Direct port of user's _2025CANDAR_SPN_variation.py
        - Algorithm: ortho/diagonal directional candidates with
          variance-weighted fusion.
        - Numba JIT used for inner fusion loop only (calculate_candidate
          uses Python dicts, kept in Python for clarity/correctness).
    """
    img = noisy.astype(np.float64)
    h, w = img.shape
    configs = _candar_get_configs()
    pad_size = window_size // 2

    for it in range(max_iter):
        noise_locations = np.argwhere((img == 0.0) | (img == 255.0))
        if noise_locations.size == 0:
            break

        # 1. Calculate candidates (Python level - uses dicts)
        cand_ortho_arr = np.full(img.shape, -1.0, dtype=np.float64)
        cand_diag_arr = np.full(img.shape, -1.0, dtype=np.float64)
        for r, c in noise_locations:
            cand_ortho_arr[r, c] = _candar_calculate_candidate(
                img, r, c, h, w, configs['ortho']
            )
            cand_diag_arr[r, c] = _candar_calculate_candidate(
                img, r, c, h, w, configs['diagonal']
            )

        img_next = img.copy()

        # 2. Fusion + update (JIT)
        changed_count = _candar_fuse_jit(
            img, img_next, noise_locations,
            cand_ortho_arr, cand_diag_arr, pad_size
        )

        img = img_next
        if changed_count == 0:
            break

    return img.clip(0, 255).astype(np.uint8)


# ============================================================================
# NEWMMF (PROPOSED, this paper)
#
# Reference: Kong & Choi (to be submitted to IEEE Access).
# Implementation: direct port of user's reference code
#                 260519_final_method_gray_.py
#
# Algorithm:
#   Phase 1 (S1): Cascading Von Neumann weighted mean iteration.
#     - Try radius r=1 (need >=2 clean values), then r=2 (>=2), then r=3 (>=3).
#     - Manhattan-distance weighting: w = 1 / Manhattan_distance.
#     - Reflect boundary.
#     - Iterate until no further changes or MAX_ITER_S1=80 reached.
#
#   Phase 2 (S2): 3x3 Von Neumann (4-neighbor) median refinement.
#     - 1 pass, applies to every pixel in noise_mask (var_thr=0).
#     - Reflect boundary.
#
# Notes:
#   - Phase 1 is byte-for-byte identical to the reference code
#     260519_final_method_gray_.py. Phase 2 has been changed from the
#     reference's 3x3 Moore median to a 3x3 Von Neumann (4-neighbor)
#     median for this variant. numba JIT is used (matching the reference).
#   - Noise candidates are detected blindly from the noisy image as the set
#     of pixels equal to 0 or 255, so the same detection rule is shared with
#     every baseline (no ground-truth mask is used).
# ============================================================================

MAX_ITER_S1 = 80


def _build_von(r):
    """Build Von Neumann neighborhood offsets up to Manhattan distance r,
    with weight 1/distance.  Identical to NEWMMF reference _build_von."""
    offsets = []
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            md = abs(dx) + abs(dy)
            if md <= r:
                offsets.append((dx, dy, 1.0 / md))
    return (
        np.array([o[0] for o in offsets], np.int64),
        np.array([o[1] for o in offsets], np.int64),
        np.array([o[2] for o in offsets], np.float64),
    )


_V1_DR, _V1_DC, _V1_WT = _build_von(1)
_V2_DR, _V2_DC, _V2_WT = _build_von(2)
_V3_DR, _V3_DC, _V3_WT = _build_von(3)


@njit(cache=True)
def _newmmf_phase1(ch_img, noise_mask, h, w):
    """
    NEWMMF Phase 1: Cascading Von Neumann weighted mean iteration.
    Identical to NEWMMF reference _s1_tracked (without stats tracking).
    """
    img = ch_img.copy()
    vals = np.empty(64, np.float64)
    wts = np.empty(64, np.float64)

    for it in range(MAX_ITER_S1):
        # Check if any noise pixels remain
        has_n = False
        for i in range(h):
            for j in range(w):
                if noise_mask[i, j] and (img[i, j] == 0.0 or img[i, j] == 255.0):
                    has_n = True
                    break
            if has_n:
                break
        if not has_n:
            break

        nxt = img.copy()
        cnt_total = 0

        for x in range(h):
            for y in range(w):
                if not noise_mask[x, y]:
                    continue
                if img[x, y] != 0.0 and img[x, y] != 255.0:
                    continue

                found = False
                cc = 0

                # r=1 attempt
                for kk in range(_V1_DR.shape[0]):
                    nx = reflect_index(x + _V1_DR[kk], h)
                    ny = reflect_index(y + _V1_DC[kk], w)
                    v = img[nx, ny]
                    if not noise_mask[nx, ny] or is_clean(v):
                        vals[cc] = v
                        wts[cc] = _V1_WT[kk]
                        cc += 1
                if cc >= 2:
                    found = True

                # r=2 attempt
                if not found:
                    cc = 0
                    for kk in range(_V2_DR.shape[0]):
                        nx = reflect_index(x + _V2_DR[kk], h)
                        ny = reflect_index(y + _V2_DC[kk], w)
                        v = img[nx, ny]
                        if not noise_mask[nx, ny] or is_clean(v):
                            vals[cc] = v
                            wts[cc] = _V2_WT[kk]
                            cc += 1
                    if cc >= 2:
                        found = True

                # r=3 attempt
                if not found:
                    cc = 0
                    for kk in range(_V3_DR.shape[0]):
                        nx = reflect_index(x + _V3_DR[kk], h)
                        ny = reflect_index(y + _V3_DC[kk], w)
                        v = img[nx, ny]
                        if not noise_mask[nx, ny] or is_clean(v):
                            vals[cc] = v
                            wts[cc] = _V3_WT[kk]
                            cc += 1
                    if cc >= 3:
                        found = True

                if not found:
                    continue

                # Weighted mean
                sw = 0.0
                swv = 0.0
                for i2 in range(cc):
                    sw += wts[i2]
                    swv += wts[i2] * vals[i2]
                nxt[x, y] = swv / sw
                cnt_total += 1

        img = nxt
        if cnt_total == 0:
            break

    return img


@njit(cache=True)
def _newmmf_phase2(img, noise_mask, h, w):
    """
    NEWMMF Phase 2: 3x3 Von Neumann median refinement.
    1 pass, applied to every noise pixel (var_thr=0).
    Uses the 4 orthogonal (Von Neumann r=1) neighbors; center excluded.
    (Changed from the reference's Moore 8-neighbor median.)
    """
    result = img.copy()
    vals = np.empty(4, np.float64)
    for x in range(h):
        for y in range(w):
            if not noise_mask[x, y]:
                continue
            cc = 0
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if abs(dr) + abs(dc) != 1:
                        continue
                    nx = reflect_index(x + dr, h)
                    ny = reflect_index(y + dc, w)
                    vals[cc] = result[nx, ny]
                    cc += 1
            result[x, y] = median_array(vals, cc)
    return result


def run_newmmf(noisy, noise_mask=None, return_phases=False):
    """
    Public API for NEWMMF (proposed method).

    Args:
        noisy:         2D array, SPN-corrupted image.
        noise_mask:    optional boolean array of noise candidates. If None
                       (default), candidates are detected blindly from the
                       noisy image as the pixels equal to 0 or 255 — the same
                       detection rule used by every baseline. A mask may still
                       be passed for backward compatibility.
        return_phases: if True, return (phase1_result, phase2_result) as a
                       tuple so the two stages can be evaluated separately.
                       Default False keeps the original behavior (final result
                       only), so existing callers are unaffected.

    Returns:
        - return_phases=False (default): final Phase 2 image as float64.
        - return_phases=True:            (s1, s2) tuple, both float64, where
                                         s1 is the Phase 1 output and s2 is the
                                         Phase 2 (final) output.
        Caller may clip and cast to uint8.
    """
    h, w = noisy.shape
    img_f = noisy.astype(np.float64)
    if noise_mask is None:
        noise_mask = (img_f == 0.0) | (img_f == 255.0)
    s1 = _newmmf_phase1(img_f, noise_mask, h, w)
    s2 = _newmmf_phase2(s1, noise_mask, h, w)
    if return_phases:
        return s1, s2
    return s2