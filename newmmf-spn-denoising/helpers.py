#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
helpers.py — Numba JIT helpers shared by all method implementations.

These mirror the helpers in NEWMMF reference (260519_final_method_gray_.py)
so that all baseline implementations use identical primitives:

  - reflect_index:  boundary reflection (matches NEWMMF _ref)
  - is_noisy:       True if pixel value is 0 or 255
  - is_clean:       True if pixel value is neither 0 nor 255
  - median_array:   insertion-sort median (matches NEWMMF _median_arr)
"""

import numpy as np
from numba import njit


@njit(cache=True, inline='always')
def reflect_index(idx, size):
    """
    Reflect boundary index.

    Maps:
      -1 -> 0, -2 -> 1, ..., size -> size-1, size+1 -> size-2, ...

    Identical to NEWMMF reference (_ref).
    """
    if idx < 0:
        return -idx - 1
    elif idx >= size:
        return 2 * size - 1 - idx
    return idx


@njit(cache=True, inline='always')
def is_noisy(v):
    """True if v is salt (255) or pepper (0)."""
    return v == 0.0 or v == 255.0


@njit(cache=True, inline='always')
def is_clean(v):
    """True if v is neither salt nor pepper."""
    return v != 0.0 and v != 255.0


@njit(cache=True)
def median_array(a, n):
    """
    Median of first n elements of array `a` via insertion sort on copy.

    Identical to NEWMMF reference (_median_arr).
    Numba-JIT compiled for speed.

    Args:
        a: 1D float64 numpy array (length >= n)
        n: number of elements to consider (a[0..n-1])

    Returns:
        median value (float)
    """
    b = np.empty(n, np.float64)
    for i in range(n):
        b[i] = a[i]
    for i in range(1, n):
        k = b[i]
        j = i - 1
        while j >= 0 and b[j] > k:
            b[j + 1] = b[j]
            j -= 1
        b[j + 1] = k
    return b[n // 2] if n % 2 == 1 else 0.5 * (b[n // 2 - 1] + b[n // 2])
