# openSMILE baseline and two data issues (Efe, 22 November)

## Two findings that affect everyone

### 1. Valence and arousal are swapped in the 2013 static labels

For the 744 clips of the 2013 batch, the static `valence_mean` column actually holds arousal and `arousal_mean` holds valence. The 2014 batch is consistent. Evidence: `data/audit/va_swap_report.md`; script: `data/audit/check_va_swap.py`.

| batch | r(static V, dynamic V) | r(static V, dynamic A) | r(static A, dynamic A) | r(static A, dynamic V) |
|---|---|---|---|---|
| 2013 | 0.575 | **0.956** | 0.570 | **0.949** |
| 2014 | **0.912** | 0.591 | **0.942** | 0.569 |

- The static labels were compared with the time average of the same clips' dynamic (2 Hz) annotations.
- Loudness (openSMILE RMS energy) points the same way. In 2014, energy relates more strongly to arousal (ρ 0.42 vs 0.32). With the raw 2013 labels it relates more strongly to valence (0.54 vs 0.32); after swapping, the pattern matches 2014.
- Neither the DEAM manual nor a web search mentions an erratum for this.
- The per-rater static file agrees with the averaged file, so it carries the same swap.

**What to do:**
- All models should use `data/interim/labels_fixed.csv`. It is built from the audit's `labels.csv` with:
  `python data/audit/check_va_swap.py --raw data/raw --fix`
  The command swaps the V and A columns (μ, s, raw, file, adj) back for the 2013 batch and adds a `va_swap_fixed` flag.
- **Emir:** the `quadrant` column of `folds_artist.csv` was computed from the swapped labels, so quadrants 2 and 4 are exchanged for 2013 clips. The artist grouping is still valid; only the stratification of the 2013 part is slightly off. Your call whether to regenerate the file or note this in the report.
- **Cem:** the prototype heteroscedastic results were produced with the swapped labels and need to be rerun with the fixed ones.
- All prototype results in the repo (`outputs/`) carry this error.

### 2. The 2013 openSMILE files cover the full song

- For 2013 clips, `features/<id>.csv` covers the full song (up to 585 s), not the 45 s excerpt. The audio file and the ratings belong to the 45 s excerpt.
- `features/aggregate_opensmile.py` now keeps only the excerpt, using the segment start in `metadata_2013.csv`. The format is min.sec: `1.29` = 89 s, `0.2` = 20 s.
- The excerpt position was verified on 21 clips by matching the mp3 excerpt's energy curve to the full-song openSMILE energy; all matched within 0.5 s.
- **Can:** MERT runs on the 45 s mp3 itself, so it is not affected.

## Baseline results (fixed labels, cropped features)

Setup:
- 520-d openSMILE (mean and std of 260 descriptors over the excerpt).
- Hyperparameters chosen per fold and per dimension on validation RMSE; the model is refit on the 8 training folds.
- Predictions are in `results/predictions/` in the contract format.

Mean ± SD over the 10 test folds:

| Model | split | R²_V | R²_A | CCC_V | CCC_A | RMSE_V | RMSE_A |
|---|---|---|---|---|---|---|---|
| Ridge | artist | 0.372 ± 0.079 | 0.671 ± 0.061 | 0.563 | 0.806 | 0.216 | 0.189 |
| SVR (RBF) | artist | 0.379 ± 0.075 | 0.693 ± 0.045 | 0.565 | 0.822 | 0.215 | 0.183 |
| Ridge | random | 0.395 ± 0.046 | 0.677 ± 0.036 | 0.579 | 0.810 | 0.215 | 0.189 |
| SVR (RBF) | random | 0.414 ± 0.047 | 0.698 ± 0.031 | 0.599 | 0.823 | 0.211 | 0.183 |

- **C1 (sanity check):** consistent with the literature (Cheuk: 0.32 / 0.64) and above the proposal threshold (0.30 / 0.60).
- **SVR vs. ridge:** SVR is significantly better on arousal (8/10 folds, ΔR² = +0.022, Wilcoxon p = 0.049). No difference on valence (5/10 folds).
- **Artist effect:** the artist-grouped split is slightly harder than the random split. For SVR, ΔR² ≈ 0.035 on valence and ≈ 0.005 on arousal. The instructor's warning that thresholds are optimistic under artist-disjoint evaluation holds for valence, not for arousal.
- **By batch (SVR, artist):** 2013: R² 0.37 / 0.65. 2014: 0.43 / 0.74. Before the fix, 2013 arousal was 0.29.
- **Constant-σ reference (`*_homo`, σ = validation RMSE):** PICP68 0.67–0.68, PICP95 0.95. It is already calibrated on average, so the heteroscedastic head has to beat it on NLL and on σ–error ranking.

Effect of the fixes (ridge, pooled OOF R², V / A):

| condition | V | A |
|---|---|---|
| data as supplied | 0.46 | 0.50 |
| excerpt cropping | 0.51 | 0.51 |
| cropping and label fix | 0.38 | 0.68 |

The high valence before the fix came from predicting arousal under the name "valence" for the 2013 clips.

## Files

| File | What it does |
|---|---|
| `features/aggregate_opensmile.py` | openSMILE cache (with cropping) |
| `data/audit/check_va_swap.py`, `va_swap_report.md` | V/A swap check and fix |
| `models/baseline_opensmile.py` | Ridge, SVR and the `*_homo` reference |
| `eval/evaluate_v1.py` | Written to compute the baseline metrics. **Emir, this is your area**: use it as a starting point or replace it with your script. |
| `splits/folds_random.csv` | Random 10-fold (StratifiedKFold on quadrant, seed 415). Emir's area; contract format, please review. |
| `results/predictions/opensmile_*` | Baseline predictions (artist and random) |
| `results/ablation_*`, `results/label_swap_check/` | Ablations and pre-fix results |
| `report/` | Report drafts: related work, deviation table, refs.bib, figures, tables |
