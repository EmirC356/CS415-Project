# Shared data contracts

Everything a workstream hands to another goes through one of these files. Change a contract only by a PR that everyone has seen.

## Label file (of record)

**Path:** `data/interim/labels_fixed.csv` (local, git-ignored). Every model, metric and figure uses this file.

How it is built:

```bash
python data/audit/audit_deam.py --raw data/raw
python data/audit/check_va_swap.py --raw data/raw --fix --labels data/interim/labels.csv --out data/interim/labels_fixed.csv
```

- **`labels.csv`:** the audit output. It keeps the orientation DEAM supplies, in which the 2013 static valence and arousal are swapped.
- **`labels_fixed.csv`:** swaps the 2013 batch back (744 clips, flag `va_swap_fixed`). The swap is detected against the time-averaged dynamic annotations.
- **Rater-adjusted s:** `s_V_adj` and `s_A_adj` are computed by the audit in corrected orientation, so they are correct in `labels_fixed.csv`.
- **Do not train on `labels.csv`.** It is only an intermediate file.
- **Folds:** the fold file was stratified on the quadrants of `labels.csv`, i.e. before the swap was found. For the 2013 clips, quadrants 2 and 4 are therefore exchanged in the stratification key. Integrity is unaffected: no fold's quadrant share deviates by more than 0.10 under either label version. The folds are kept as they are, so that all results stay comparable.

## Fold file

**Owner:** Emir. **Path:** `splits/folds_artist.csv` (artist-grouped). A second file, `splits/folds_random.csv`, holds the random comparison and is made after the report.

| Column | Type | Meaning |
|---|---|---|
| `song_id` | int | DEAM song id |
| `artist_norm` | str | Normalised artist name (the grouping key) |
| `quadrant` | int 1–4 | V–A quadrant of the rescaled μ (the stratification key): 1 = V≥0, A≥0; 2 = V<0, A≥0; 3 = V<0, A<0; 4 = V≥0, A<0. A μ of exactly 0 counts as positive (52 clips for V, 51 for A). |
| `fold` | int 0–9 | Test fold of this clip |

**Construction:** `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=415)`, with groups `artist_norm` and y `quadrant`. Rows are sorted by `song_id` before splitting. Script: `splits/make_folds.py`; settings and per-fold shares: `splits/folds_artist_report.md`.

**Validation inside a fold:** for test fold `k`, fold `(k + 1) % 10` is the validation set and the other 8 folds are training (≈ 80/10/10). Everyone uses this rule.

**Integrity test, which must pass before anyone uses the file:**
- Every clip appears exactly once.
- No artist appears in two folds.
- Quadrant shares per fold are reported.

If the artist field is missing or unusable, group by `song_id` instead and record this as a deviation.

## Embedding cache

**Owner:** Can. **Path:** `cache/<encoder>/` (git-ignored, shared out-of-band).

| Encoder | File | Shape |
|---|---|---|
| openSMILE | `cache/opensmile/clip_features.parquet` | clip × (mean, std of each feature) |
| MERT-v1-95M | `cache/mert95m/<song_id>.npy` | window × 13 layers × 768; 5 s windows at 24 kHz |

- Layer pooling (window mean) happens in the model code, not in the cache.
- Record the extraction settings in `cache/<encoder>/meta.json`.

## Prediction file

**Owner:** Emir. **Path:** `results/predictions/<model>__<split_type>__seed<k>.csv` (committed).

| Column | Type | Meaning |
|---|---|---|
| `fold_id` | int 0–9 | Test fold this prediction belongs to |
| `split_type` | str | `artist` or `random` |
| `seed` | int | Training seed |
| `model` | str | e.g. `opensmile_ridge`, `mert_ridge_L7`, `mert_hetero_b0.5` |
| `song_id` | int | DEAM song id |
| `mu_V`, `mu_A` | float | Predicted mean, rescaled [−1, 1] |
| `sigma_V`, `sigma_A` | float | Predicted σ in the same units; empty (NaN) for point models |

**Rules:**
- Write only test-fold predictions. Each clip appears once per (model, split_type, seed).
- Selected hyperparameters and layers go in a sidecar file, `<same name>.json`.
