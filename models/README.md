# models/ (owners: Efe for the baseline, Cem for the heads)

Every model reads the fold file plus one cache and writes one prediction file per (split_type, seed). The format is in [`docs/contracts.md#prediction-file`](../docs/contracts.md#prediction-file).

## Heads

| Name | Input | Notes |
|---|---|---|
| `opensmile_ridge`, `opensmile_svr` | openSMILE | Baseline. Hyperparameters are chosen on the validation fold. |
| `mert_ridge_L*` | MERT window-mean, one layer | Headline result. The layer is chosen on the validation fold. |
| `mert_mlp` | MERT | One hidden layer (512), MSE loss |
| `opensmile_mlp` | openSMILE | Same head, used for the same-head comparison (C2) |
| `*_hetero_b0.5`, `*_hetero_b0` | Same inputs | μ and σ heads with β-NLL. β = 0.5 is primary; β = 0 is plain NLL. |
| `*_homo` | Same μ̂ | Constant σ fitted on **validation** residuals |

## Variance stability (heteroscedastic head)

- Soft bounds: `sigma = s_min + (s_max - s_min) * sigmoid(z)`.
  - `s_min = 0.01`.
  - `s_max = 2 * std(y_train)`, computed per dimension from training data only.
  - No hard clamp, because it zeroes the gradient.
- Warm-up: for the first epochs, train μ with MSE only, then switch to β-NLL.
- Use weight decay, and early-stop on **plain** validation NLL. Select models on plain NLL and RMSE, never on the β-NLL loss value (Seitzer et al. 2022, §4.1).
- Log per epoch: σ percentiles (1, 5, 50, 95, 99), the share of σ within 1% of either bound, plain NLL and RMSE.
- If more than 5% of validation σ sit at a bound, report it as a stability issue. Do not change the method after looking at test results.

## openSMILE baseline (implemented)

```bash
python models/baseline_opensmile.py --labels data/interim/labels_fixed.csv                       # artist folds
python models/baseline_opensmile.py --labels data/interim/labels_fixed.csv \
    --folds splits/folds_random.csv --split-type random                                         # random folds
```

- Heads: ridge (α ∈ logspace(−1, 5, 25)) and RBF-SVR (C ∈ {0.1, 0.3, 1, 3, 10}, γ = {0.25, 0.5, 1, 2}/d, ε ∈ {0.02, 0.05, 0.1}).
- Per fold and dimension: standardise on train, choose on validation RMSE, refit on the 8 training folds, predict the test fold once. Choices are in the `.json` sidecar.
- Also writes `*_homo` files: the same μ with a constant σ per fold and dimension = validation RMSE (the homoscedastic reference for C5).
- **Use `labels_fixed.csv`**, see `data/audit/va_swap_report.md`.
