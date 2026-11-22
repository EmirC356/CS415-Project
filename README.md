# CS415 Group 21: Uncertainty-Aware Music Emotion Recognition on DEAM

This is the CS415 (Intro to Deep Learning, Fall 2026–27) group project of Can Yüzbey, Cem Özkul, Efe Kumru and Emir Ceylan.

**Task.** We predict static valence and arousal for the 1,744 45-second DEAM clips from 2013–2014. We use frozen pretrained audio encoders (MERT-v1-95M first, PANNs later) and compare them with an openSMILE baseline. A heteroscedastic regression head predicts an uncertainty σ for each clip.

**What σ means in this project:**

> σ estimates the predictive error of the *mean* rating. It covers model approximation error plus the averaging noise in the mean rating (≈ s²/n). It is **not** the spread of individual listeners. A single network's σ also leaves out epistemic uncertainty, which we look at through the spread across a seed ensemble.

The full research framework (hypotheses, criteria, protocol, roles) is in [`docs/framework.md`](docs/framework.md).

## Progress report scope (due 2026-11-22)

The order follows the instructor's feedback:

1. Data and split integrity: annotations, artist identifiers, and zero artist overlap across folds.
2. openSMILE baseline (ridge and SVR).
3. Frozen MERT with a simple head (ridge, then MLP), compared with openSMILE **using the same head**. RMSE, R², CCC and Pearson are reported separately for valence and arousal.
4. Heteroscedastic head: calibration, correlation of σ with the actual error, and a descriptive relation to annotator std. Variance stability diagnostics are logged.

Deferred until after the report: PANNs, randomly initialised MERT, the layer probe (H2b) and LP-FT.

## Repository layout

```
data/            how to obtain DEAM; raw data stays local (git-ignored)
splits/          committed fold files (artist-grouped, quadrant-stratified 10-fold)
features/        feature / embedding extraction code; caches go to cache/ (ignored)
models/          regression heads: baseline, ridge/MLP, heteroscedastic
eval/            shared metrics script; reads prediction files only
results/
  predictions/   one CSV per model run, in the shared format (committed)
docs/            framework, data contracts
```

Each folder has a short README naming its owner and its input/output contract.

## Shared contracts

Workstreams connect only through three files:

| Contract | Spec | Owner |
|---|---|---|
| Fold file | [`docs/contracts.md#fold-file`](docs/contracts.md#fold-file) | Emir |
| Embedding cache | [`docs/contracts.md#embedding-cache`](docs/contracts.md#embedding-cache) | Can |
| Prediction file | [`docs/contracts.md#prediction-file`](docs/contracts.md#prediction-file) | Emir |

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Then follow [`data/README.md`](data/README.md) to download DEAM.

## Working rules

- Work on a branch and open a PR into `main`. Do not push directly to `main`.
- Never commit DEAM audio, annotations or caches. The data licence is CC BY-NC, and the files are large.
- Choose hyperparameters and layers on each fold's validation part only. Never tune on test folds.
- Every model writes a prediction file. All metrics come from the single script in `eval/`.
- Fix the seeds and record them in the prediction file.

---

## Legacy prototype (`src/`, pre-contract)

> **Not the contract protocol.** `src/`, `outputs/`, the root `splits.csv` and `analysis_report.ipynb` are the pre-contract prototype (`GroupKFold`, seed 42, uncorrected labels). Their numbers are **not comparable** to the contract protocol in [`docs/contracts.md`](docs/contracts.md) and must not be cited in the report.
>
> The heteroscedastic head on the contract protocol lives in `src/train.py` on branch `cem/hetero-corrected-labels` (to be merged).
