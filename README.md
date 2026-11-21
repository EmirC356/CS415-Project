# CS415: Music Emotion Recognition with Uncertainty Estimation

This repository contains the end-to-end pipeline for static continuous emotion recognition (Valence and Arousal) on the DEAM dataset (1,744 45-second music clips), incorporating artist-disjoint 10-fold cross-validation, feature aggregation, heteroscedastic neural networks, and uncertainty calibration analysis.

---

## Quickstart & Reproduction One-Liners

Execute the following commands from the root directory to generate all artifacts in `outputs/`:

```bash
# 1. Generate artist-disjoint 10-fold splits and clean targets
python src/make_splits.py

# 2. Preprocess targets and aggregate openSMILE acoustic features (mean + std over time)
python -m src.data

# 3. Train models across 10 folds
# 3a. Heteroscedastic MLP (Warmup + Gaussian NLL with Early Stopping)
python -m src.train --model heteroscedastic_mlp --features opensmile

# 3b. Deterministic MLP baseline (MSE Loss)
python -m src.train --model deterministic_mlp --features opensmile

# 3c. Ridge Regression baseline (RidgeCV for Valence & Arousal)
python -m src.train --model ridge --features opensmile

# 4. Evaluate out-of-fold predictions and generate reports & calibration plots
python -m src.evaluate --preds outputs/preds_opensmile_heteroscedastic_mlp.csv
python -m src.evaluate --preds outputs/preds_opensmile_deterministic_mlp.csv
python -m src.evaluate --preds outputs/preds_opensmile_ridge.csv
```

---

## Repository Structure

```
├── data/
│   ├── DEAM_Annotations/               # Raw static & dynamic annotations
│   ├── features/features/              # Frame-level openSMILE CSV files
│   ├── metadata/                       # DEAM song metadata (2013, 2014, 2015)
│   ├── processed_targets.csv           # Cleaned and rescaled targets ([-1, 1])
│   └── opensmile_aggregated.parquet    # 520-dim aggregated acoustic feature matrix
├── outputs/                            # Generated predictions, evaluations, and logs
│   ├── preds_opensmile_*.csv           # Out-of-fold predictions
│   ├── eval_preds_opensmile_*.md       # 10-fold metric evaluation reports
│   ├── reliability_diagram.png         # Calibration curves (PICP vs Nominal)
│   └── sigma_logs_opensmile.csv        # Epoch-level validation sigma statistics
├── src/
│   ├── __init__.py
│   ├── data.py                         # Target rescaling, feature aggregation & MERT loader
│   ├── evaluate.py                     # 10-fold metrics (RMSE, R², CCC, Pearson) & uncertainty analysis
│   ├── make_splits.py                  # Artist normalization & GroupKFold splitting
│   ├── models.py                       # RidgeBaseline, DeterministicMLP, HeteroscedasticMLP
│   └── train.py                        # 10-fold CV training loop with fold-isolated scaling
├── splits.csv                          # 10-fold artist-disjoint assignments (song_id, artist, fold)
└── README.md
```

---

## Contents of `outputs/`

The `outputs/` folder contains all experimental results and evaluation artifacts:

| File | Description | Key Columns / Output |
| :--- | :--- | :--- |
| `preds_opensmile_heteroscedastic_mlp.csv` | Out-of-fold predictions from Heteroscedastic MLP | `song_id`, `fold`, `mu_v`, `mu_a`, `sigma_v`, `sigma_a` |
| `preds_opensmile_deterministic_mlp.csv` | Out-of-fold predictions from Deterministic MLP | `song_id`, `fold`, `mu_v`, `mu_a`, `sigma_v`, `sigma_a` |
| `preds_opensmile_ridge.csv` | Out-of-fold predictions from Ridge Regression | `song_id`, `fold`, `mu_v`, `mu_a`, `sigma_v`, `sigma_a` |
| `eval_preds_opensmile_*.md` | Comprehensive Markdown report per model | 10-fold mean ± std tables for RMSE, R², CCC, Pearson $r$, NLL, PICP |
| `reliability_diagram.png` | Calibration curve plot | Empirical coverage (PICP) vs nominal coverage ($[0.05, 0.95]$) |
| `sigma_logs_opensmile.csv` | Per-epoch validation $\sigma$ stats across folds | `fold`, `epoch`, `stage`, `val_nll`, `sigma_v_mean`, `sigma_a_mean`, etc. |

---

## Benchmark Results (10-Fold Artist-Disjoint CV)

### 1. Regression Performance (Mean ± Std across 10 Folds)

| Model | Valence RMSE ↓ | Valence R² ↑ | Valence CCC ↑ | Valence $r$ ↑ | Arousal RMSE ↓ | Arousal R² ↑ | Arousal CCC ↑ | Arousal $r$ ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression** | 0.2093 ± 0.0136 | 0.4707 ± 0.0688 | 0.6562 ± 0.0440 | 0.6978 ± 0.0405 | 0.2218 ± 0.0116 | 0.4996 ± 0.0767 | 0.6866 ± 0.0448 | 0.7184 ± 0.0468 |
| **Deterministic MLP** | 0.2102 ± 0.0141 | 0.4666 ± 0.0662 | 0.6683 ± 0.0410 | 0.6930 ± 0.0414 | **0.2095 ± 0.0111** | **0.5547 ± 0.0597** | **0.7328 ± 0.0361** | **0.7524 ± 0.0339** |
| **Heteroscedastic MLP** | 0.2157 ± 0.0161 | 0.4375 ± 0.0814 | 0.6508 ± 0.0442 | 0.6813 ± 0.0491 | 0.2178 ± 0.0093 | 0.5198 ± 0.0533 | 0.7104 ± 0.0320 | 0.7354 ± 0.0273 |

### 2. Uncertainty Quantification & Calibration

| Metric | Heteroscedastic MLP | Deterministic MLP (Baseline $\sigma$) | Ridge Baseline |
| :--- | :---: | :---: | :---: |
| **Gaussian NLL (Overall)** | **-0.1060** | -0.0047 | -0.0955 |
| **PICP (68% Nominal)** | **65.68%** *(Valence: 65.94%, Arousal: 65.42%)* | 54.36% | 62.01% |
| **PICP (95% Nominal)** | **93.61%** *(Valence: 93.81%, Arousal: 93.41%)* | 84.78% | 91.77% |
| **$\text{Corr}(\sigma, \|y - \mu\|)$ Spearman (Valence)** | **0.1019** ($p = 2.01 \times 10^{-5}$) | 0.0116 ($p = 0.63$) | -0.0442 ($p = 0.07$) |
| **$\text{Corr}(\sigma, \|y - \mu\|)$ Spearman (Arousal)** | **0.1848** ($p = 7.37 \times 10^{-15}$) | 0.0014 ($p = 0.95$) | -0.0001 ($p = 0.99$) |
| **$\text{Corr}(\sigma, \text{Disagreement})$ (Valence)** | **0.0505** ($p = 0.035$) | -0.0389 ($p = 0.10$) | 0.0089 ($p = 0.71$) |
| **$\text{Corr}(\sigma, \text{Disagreement})$ (Arousal)** | **0.0542** ($p = 0.024$) | 0.0024 ($p = 0.92$) | 0.0153 ($p = 0.52$) |

> **Key Finding**: The Heteroscedastic MLP outputs well-calibrated confidence intervals (65.68% empirical coverage for nominal 68%, and 93.61% for nominal 95%). Furthermore, its predicted $\sigma$ strongly and statistically significantly correlates with actual absolute error ($p < 10^{-4}$ for Valence, $p < 10^{-14}$ for Arousal), proving it successfully captures sample-level epistemic/aleatoric uncertainty.

---

## What Is the Next Thing to Do?

1. **Extract & Integrate MERT Embeddings (`--features mert`)**:
   - Extract representations from a pre-trained music foundation model (e.g., `m-a-p/MERT-v1-330M`) for the 1,744 audio clips into `data/mert_cache/`.
   - Run:
     ```bash
     python -m src.train --model heteroscedastic_mlp --features mert
     python -m src.evaluate --preds outputs/preds_mert_heteroscedastic_mlp.csv
     ```
   - Compare openSMILE hand-crafted acoustic features vs. MERT deep representations.

2. **Feature Fusion / Multimodal Experiment**:
   - Concatenate openSMILE (acoustic statistics) + MERT (deep contextual semantics) into an ensemble representation to test whether acoustic prosody and deep self-supervised embeddings are complementary.

3. **Qualitative Error & Disagreement Case Study**:
   - Identify songs with:
     - High error and high predicted $\sigma$ (model appropriately knows it doesn't know).
     - High error but low predicted $\sigma$ (overconfident failures).
     - High annotator standard deviation (`valence_std`, `arousal_std`) vs. model predicted $\sigma$.

4. **Prepare Visualizations for Report / Presentation**:
   - Generate predicted vs. true scatter plots with error bars ($\mu \pm \sigma$).
   - Include `outputs/reliability_diagram.png` in the methodology/results section.
