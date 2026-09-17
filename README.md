# Stellar Transit Drift Classification And Localization

| | |
| --- | --- |
| Final rank | 3rd |
| Domain | Computer Vision |
| Difficulty | Medium |
| Scoring | ↑ Higher is better |
| Compute | A10G |
| Challenge status | Accepted / closed |
| Solutions submitted | 4 |
| Last submission | 2026-06-26 |

## Problem statement

### Syzygy: Stellar Transit Image Recognition

### Overview

Each sample is a single-channel **40×40 grayscale image** of one star, with pixel

intensities near 1.0. The image is a brightness scan of the star arranged row-major into

the grid: pixel `(r, c)` holds scan position `40·r + c`, so flattening the image row-major

recovers the original 1600-pixel scan. You may work directly in 2-D or on the flattened

scan.

A transiting planet stamps a train of faint, **regularly spaced dark dips** along the scan,

one every `P` pixels, where the spacing `P` is given for each image. On a clean image those

dips sit on a perfect periodic lattice. In two thirds of the images, a hidden companion pulls the dips

**off the lattice by a slow sinusoid**, so the dip centers visibly drift early/late as you

move along the scan. The size of that systematic drift is what you must recognize. The dips

you observe are measured with noise — each visible dip center is scattered slightly off its

true position — so the lattice and drift can only be pinned down by combining many dips, not

one. The underlying true positions, including the target defined below, are noise-free.

Many images are also covered by **stellar-spot banding**: smooth, large-amplitude intensity

undulations that can be darker than the dips themselves. Spots distort the *background*, not

the dip *positions*. Spot banding appears in exactly half of the images in every class, so

it carries no information about the answer — it is a balanced distractor.

For each test image, report two things:

- `regime` — how far the dip centers drift off the periodic lattice, one of `{none, weak, strong}`:
   - `none` : no drift — the true dip positions lie on a plain periodic lattice.
   - `weak` : a *small* periodic drift of the true positions off the lattice.
   - `strong` : a *large* periodic drift of the true positions off the lattice.
- `next_pos` — the **true (noise-free) row-major pixel index** of the center of the next dip past the final pixel (1599). It is the lattice-plus-drift position extrapolated one step beyond the grid; the measurement noise on the visible dips is **not** added to it, so it is fully determined by the lattice and drift you recover.

The intended first-order approach is to **localize the dip centers** in the image, compare

them to the ideal lattice at spacing `P`, and read the position residuals: their amplitude

sets `regime`, and extrapolating the drift (not just the straight lattice) gives `next_pos`.

The noise on the visible dips limits how precisely you can pin the lattice and drift, which

is the main source of difficulty. A convolutional model can also be trained end-to-end.

### Evaluation

The score combines a classification term and a localization term. Let the test set have `M`

images.

**Classification term (macro-F1 over the three balanced classes).** For each class `c` in

`{none, weak, strong}` with true/false positives/negatives `TP_c`, `FP_c`, `FN_c`:

- `F1_c = 2 · TP_c / (2 · TP_c + FP_c + FN_c)`, and `F1_c = 0` if that denominator is 0.
- `MacroF1 = (F1_none + F1_weak + F1_strong) / 3`.

A predicted `regime` outside the three allowed values counts as wrong for that row.

**Localization term (position tolerance).** For image `i`, with predicted and true dip

position `p_i` and `y_i`:

- `d_i = | p_i − y_i |` (in pixels)
- `f_i = exp( − d_i / τ )`, with the fixed published constant `τ = 1.5`.
- A missing, non-numeric, NaN, or infinite `next_pos` gives `f_i = 0`.
- `Localization = (1/M) · Σ_i f_i`.

**Composite (reported 0–100, higher is better).**

- `raw = 0.5 · MacroF1 + 0.5 · Localization`
- `Score = 100 · clip(raw, 0, 1)`

The scale runs from 0 to 100, and `Score = 100` corresponds to a submission that matches the

held-out answer key exactly. The target `next_pos` is the true, noise-free position of the

next dip, so it is fully determined by the lattice and drift and is predictable in principle.

The difficulty is that the lattice and drift must be recovered from dip positions that are

individually noisy, which limits how precisely you can extrapolate — better recovery scores

strictly higher. The baselines below show where current methods land on this dataset.

**Measured reference baselines** (this dataset):

- Perfect submission (equals the held-out answer key): **100.0**
- Constant baseline (predict `none` everywhere + median `next_pos`): **~10**
- Spot-distractor heuristic (flag a companion from the strongest extra intensity period; lattice `next_pos`): **~35**
- Dip-localization model (locate dips, threshold the residual amplitude, extrapolate the lattice for `next_pos`): **~57**
- A tuned model that also fits and extrapolates the drift: **~60**.

### Dataset

- `images.npz` — two aligned arrays:
   - `ids` : string array, shape `(3000,)`, every sample_id (train and test).
   - `images` : float32 array, shape `(3000, 40, 40)`; `images[r]` is the image for `ids[r]`,

```
           intensities normalized to a baseline near 1.0.
```

- `train.csv` — labeled training rows, columns:
   - `sample_id` - string - image identifier (matches a row in `images.npz`).
   - `P` - float - dip spacing in pixels.
   - `regime` - string - training label, one of `none` / `weak` / `strong`.
   - `next_pos` - float - training label, next-dip row-major pixel index.
- `test.csv` — query rows you must predict, columns:
   - `sample_id` - string - image identifier.
   - `P` - float - dip spacing in pixels.
- `sample_submission.csv` — a correctly formatted, low-scoring example submission.

### Submission

Submit a CSV with a header and exactly these columns, in this order:

- `sample_id` - string - every id in `test.csv`, each exactly once.
- `regime` - string - one of `none`, `weak`, `strong`.
- `next_pos` - float - predicted next-dip row-major pixel index.

Example of a correctly formatted submission file:

| sample_id | regime | next_pos |

|---|---|---:|

| abcdef012345 | none | 1618.4 |

| 0123456789ab | weak | 1626.7 |

| fedcba987654 | strong | 1632.1 |

The example values are illustrative only and are not real labels.

Validity rules:

- Exactly one row per `test.csv` id (currently 660 rows); no missing, duplicate, extra, or unknown ids; no extra or reordered columns.
- The header is required and must match exactly.
- An out-of-vocabulary `regime` value is counted as an incorrect prediction (it is scored exactly like naming a wrong class — a false negative for the true class plus a false positive — so there is no "abstain" benefit). A non-numeric / NaN / infinite `next_pos` gives zero localization credit for that row. The grader never crashes on either.

### What Not to Use

- **A single-dip / single-patch classifier.** One dip cannot reveal a drift *pattern*; the signal only exists across the whole train of dips in the image.
- **Majority-class prediction.** Classes are balanced, so it floors the score (~10 total).
- **An intensity periodogram ("is there an extra period in the pixels?").** The strongest extra intensity period is usually the spot banding, which is balanced across classes and uncorrelated with the answer. It tops out around 35.
- **Straight-lattice extrapolation for `next_pos`.** It is fine for `none` images but is off by up to the full drift amplitude on `strong` images; modeling the drift sinusoid is what separates a good localization from a mediocre one.
- **Using `P` or other metadata alone.** `next_pos` depends on the dip phase, which can only be recovered from the image itself.

What is expected: locate the dip centers in the image, remove the ideal lattice at spacing

`P`, and reason about the **position residuals** — their amplitude for `regime` and their

sinusoidal extrapolation for `next_pos`.
