# NEWMMF — Training-Free Cellular-Automata Denoising of Salt-and-Pepper Noise (Grayscale)

Reproduction code for the manuscript:

> **A training-free, cellular-automata-based two-phase filter for high-density
> salt-and-pepper noise removal in grayscale images** — Gil-Tak Kong and
> Un-Sook Choi (under review, *Computers & Electrical Engineering*).
> *(Working title; update on acceptance.)*

This repository re-implements the proposed method (**NEWMMF**) together with all
baseline methods inside a single, seed-fixed evaluation pipeline, so that every
number reported in the paper can be regenerated from scratch on standard
benchmark images.

> **Scope.** All methods included here are **non-learning / training-free /
> interpretable**. No deep-learning models are used in the quantitative
> comparison.

---

## 1. Repository layout

The modules are flat (no `src/` package). `main.py` inserts its own directory
on `sys.path`, so run it from **inside this folder**.

| File | Purpose |
|---|---|
| `helpers.py` | Numba-JIT primitives: boundary reflection, insertion-sort median, salt/pepper test |
| `utils.py` | Noise generation (`add_sp_noise`), PSNR/SSIM, grayscale I/O |
| `methods_baseline.py` | DBA, ACWMF, NAFSMF, DAMF |
| `methods_proposed.py` | MTA, CANDAR (self-citations) and **NEWMMF** (proposed) |
| `visualization.py` | PSNR / SSIM curve plotting |
| `main.py` | Full evaluation pipeline (entry point) |
| `requirements.txt`, `LICENSE`, `CITATION.cff` | Environment, license, citation metadata |

All imports are flat, e.g. `from utils import ...`.

---

## 2. Requirements

Developed and tested on **Python 3.11**.

```bash
pip install -r requirements.txt
```

This installs `numpy`, `opencv-python`, `scikit-image`, `numba`, `matplotlib`.

> For an exact-reproducibility artifact, freeze your resolved versions before
> release: `pip freeze > requirements.lock.txt`.

---

## 3. Data

The benchmark images used in the paper are **included** in this repository, so
`python main.py` runs out of the box:

```
newmmf-spn-denoising/
├── images/     Lena.*  Barbara.*  Baboon.*  Peppers.*   (4 typical test images)
└── BSD68/      test001.png … test068.png                (BSD68 benchmark, 68 images)
```

These locations match the defaults in `main.py` (`IMAGE_DIR = "./images"`,
`BSD68_DIR = "./BSD68"`); change those two constants only if you move the
images. Typical images are matched by filename keyword (case-insensitive), and
extensions are auto-detected (png / jpg / tiff / bmp).

**Sources.** USC-SIPI and other standard test images; BSD68 (the 68-image test
subset of the Berkeley BSDS300). Each dataset remains under its own license.

**Grayscale conversion.** Color inputs are converted on load with OpenCV
`IMREAD_COLOR → COLOR_BGR2GRAY`, i.e. the ITU-R BT.601 luma
`0.299 R + 0.587 G + 0.114 B`. This repository fixes the conversion so the
numbers are reproducible.

---

## 4. Run

```bash
cd this-folder
python main.py
```

Outputs are written to `./results_260703` (change `OUTPUT_DIR` in `main.py`):

```
results_260703/
├── Lena/  Barbara/  Baboon/  Peppers/
│     original.png,  noisy_50.png … noisy_90.png,
│     <Method>/denoised_50.png … denoised_90.png   (one folder per method)
├── graphs/     PSNR & SSIM curves (4 images × 2  +  BSD68 × 2)
└── csv/        results_4_typical.csv,  results_BSD68_average.csv
```

> For BSD68, only the **per-density average** is stored (no per-image denoised
> output), by design.

**Approximate runtime.** A few minutes for the 4 typical images; ~30–60 min for
BSD68 (8 methods × 5 densities × 68 images). The first run is dominated by
one-time Numba JIT compilation; subsequent runs reuse the on-disk cache.

Configuration lives in constants at the top of `main.py`
(`IMAGE_DIR`, `BSD68_DIR`, `OUTPUT_DIR`, `TYPICAL_IMAGES`, `NOISE_DENSITIES`).

---

## 5. Methods

Seven methods plus one ablation. Pipeline order:
`ACWMF, DBA, NAFSMF, DAMF, MTA, CANDAR, NEWMMF-P1, NEWMMF`.

| Method | Reference | Role in this paper |
|---|---|---|
| **DBA** | Srinivasan & Ebenezer, *IEEE SPL* 14(3):189–192, 2007 | Classical high-density baseline (canonical median-of-3×3 with left/top propagation) |
| **ACWMF** | Ko & Lee (1991); Chen & Wu, *IEEE SPL*, 2001 | Classical center-weighted-median baseline |
| **NAFSMF** | Toh & Mat Isa, *IEEE SPL*, 2010 | Fuzzy switching-median baseline |
| **DAMF** | Erkan et al., *Comput. Electr. Eng.*, 2018 | Modern adaptive decision-based baseline |
| **MTA** | Kong & Choi, 2026 | Self-citation — group-based iterative-mean ancestor |
| **CANDAR** | Kong & Choi, *CANDAR* 2025 | Self-citation — directional-candidate + variance-weighted-fusion ancestor |
| **NEWMMF** | Kong & Choi (this manuscript) | **Proposed method** |
| **NEWMMF-P1** | — | **Ablation**: Phase-1-only output of NEWMMF (isolates the Phase-2 contribution) |

**NEWMMF in brief.** Phase 1 — a cascading Von Neumann weighted mean: try radius
`r = 1 → 2 → 3`, using `1 / Manhattan-distance` weights, iterated until no
candidate pixels remain. Phase 2 — a single 3×3 Von Neumann (4-neighbour)
median refinement pass. Boundaries use index reflection.

---

## 6. Noise model

Every method is evaluated on the **same** salt-and-pepper corruption
(`utils.add_sp_noise`), a per-pixel Bernoulli model:

- With a fixed seed (`42`), draw a uniform field `U ∼ Uniform[0,1)` and a coin
  field over the image.
- A pixel is corrupted where `U < ratio`; a corrupted pixel is set to `0`
  (pepper) or `255` (salt) with equal probability (0.5 / 0.5).

Because the seed is fixed and the uniform field is drawn identically for every
density, the corrupted set at density `d` is **nested** inside the set at any
higher density — so the five density levels are directly comparable on the same
image.

Densities evaluated: **50, 60, 70, 80, 90 %** (high-density focus).

---

## 7. Reproducibility and comparison protocol

- **Author re-implementation.** Every method in this repository is
  re-implemented by the authors. **No PSNR/SSIM value is copied from any
  original publication**; all numbers in the paper are produced by this code.
- **Identical inputs.** Every method receives the same noisy image (same seed)
  and is scored with the same metrics.
- **Deterministic.** Re-running `python main.py` on the same images reproduces
  the CSV files exactly.

---

## 8. Metrics

- **PSNR** `= 10·log₁₀(255² / MSE)` (`utils.gray_psnr`; returns `100 dB` when
  `MSE = 0`).
- **SSIM** via scikit-image `structural_similarity`, `data_range = 255`
  (`utils.gray_ssim`). The restored image is clipped to `[0, 255]` and cast to
  `uint8` before scoring.

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: numba` / `cv2` | `pip install -r requirements.txt` |
| Very slow first run | One-time Numba JIT compilation; later runs use the on-disk cache |
| A typical image is skipped | Its filename must contain the keyword `Lena` / `Barbara` / `Baboon` / `Peppers` (case-insensitive), or set the exact filename in `TYPICAL_IMAGES` in `main.py` |
| BSD68 skipped | `BSD68/` directory is empty or missing |

---

## License and citation

Released under the MIT License (see `LICENSE`). If you use this code, please
cite the accompanying paper (see `CITATION.cff`).
