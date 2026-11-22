# Metrics (eval/evaluate_v1.py)

Inputs: `results/predictions/opensmile_ridge__artist__seed0.csv`, `results/predictions/opensmile_ridge_homo__artist__seed0.csv`, `results/predictions/opensmile_svr__artist__seed0.csv`, `results/predictions/opensmile_svr_homo__artist__seed0.csv`

## Per-fold mean ± SD (10 test folds)

| model | split | dim | R² | RMSE | RMSE (1–9) | CCC | Pearson r | PICP68 | PICP95 | NLL |
|---|---|---|---|---|---|---|---|---|---|---|
| opensmile_ridge | artist | A | 0.502 ± 0.083 | 0.225 ± 0.020 | 0.899 ± 0.078 | 0.686 ± 0.060 | 0.715 ± 0.057 | — | — | — |
| opensmile_ridge | artist | V | 0.495 ± 0.053 | 0.206 ± 0.012 | 0.823 ± 0.047 | 0.681 ± 0.031 | 0.714 ± 0.036 | — | — | — |
| opensmile_ridge_homo | artist | A | 0.502 ± 0.083 | 0.225 ± 0.020 | 0.899 ± 0.078 | 0.686 ± 0.060 | 0.715 ± 0.057 | 0.685 ± 0.054 | 0.944 ± 0.025 | -0.063 ± 0.105 |
| opensmile_ridge_homo | artist | V | 0.495 ± 0.053 | 0.206 ± 0.012 | 0.823 ± 0.047 | 0.681 ± 0.031 | 0.714 ± 0.036 | 0.672 ± 0.054 | 0.948 ± 0.019 | -0.158 ± 0.057 |
| opensmile_svr | artist | A | 0.529 ± 0.061 | 0.219 ± 0.017 | 0.876 ± 0.067 | 0.700 ± 0.048 | 0.733 ± 0.044 | — | — | — |
| opensmile_svr | artist | V | 0.506 ± 0.052 | 0.203 ± 0.011 | 0.814 ± 0.042 | 0.685 ± 0.040 | 0.719 ± 0.038 | — | — | — |
| opensmile_svr_homo | artist | A | 0.529 ± 0.061 | 0.219 ± 0.017 | 0.876 ± 0.067 | 0.700 ± 0.048 | 0.733 ± 0.044 | 0.672 ± 0.058 | 0.941 ± 0.026 | -0.089 ± 0.091 |
| opensmile_svr_homo | artist | V | 0.506 ± 0.052 | 0.203 ± 0.011 | 0.814 ± 0.042 | 0.685 ± 0.040 | 0.719 ± 0.038 | 0.657 ± 0.054 | 0.943 ± 0.020 | -0.168 ± 0.054 |

## Median over folds

|                                         |    R2 |   RMSE |   RMSE_19 |   CCC |     r |   PICP68 |   PICP95 |     NLL |
|:----------------------------------------|------:|-------:|----------:|------:|------:|---------:|---------:|--------:|
| ('opensmile_ridge', 'artist', 'A')      | 0.524 |  0.222 |     0.889 | 0.7   | 0.729 |  nan     |  nan     | nan     |
| ('opensmile_ridge', 'artist', 'V')      | 0.498 |  0.207 |     0.827 | 0.691 | 0.714 |  nan     |  nan     | nan     |
| ('opensmile_ridge_homo', 'artist', 'A') | 0.524 |  0.222 |     0.889 | 0.7   | 0.729 |    0.679 |    0.955 |  -0.083 |
| ('opensmile_ridge_homo', 'artist', 'V') | 0.498 |  0.207 |     0.827 | 0.691 | 0.714 |    0.676 |    0.952 |  -0.152 |
| ('opensmile_svr', 'artist', 'A')        | 0.535 |  0.22  |     0.879 | 0.7   | 0.747 |  nan     |  nan     | nan     |
| ('opensmile_svr', 'artist', 'V')        | 0.506 |  0.2   |     0.802 | 0.692 | 0.718 |  nan     |  nan     | nan     |
| ('opensmile_svr_homo', 'artist', 'A')   | 0.535 |  0.22  |     0.879 | 0.7   | 0.747 |    0.667 |    0.945 |  -0.088 |
| ('opensmile_svr_homo', 'artist', 'V')   | 0.506 |  0.2   |     0.802 | 0.692 | 0.718 |    0.645 |    0.941 |  -0.183 |

## Pooled out-of-fold (all clips)

| model                | split_type   |   seed | dim   |    n |     R2 |   RMSE |   RMSE_19 |    CCC |      r |   PICP68 |   PICP95 |      NLL |   rho_sigma_abs_err |    rho_p |
|:---------------------|:-------------|-------:|:------|-----:|-------:|-------:|----------:|-------:|-------:|---------:|---------:|---------:|--------------------:|---------:|
| opensmile_ridge      | artist       |      0 | V     | 1744 | 0.5052 | 0.2064 |    0.8256 | 0.6869 | 0.7126 | nan      | nan      | nan      |            nan      | nan      |
| opensmile_ridge      | artist       |      0 | A     | 1744 | 0.5082 | 0.2261 |    0.9043 | 0.6899 | 0.7148 | nan      | nan      | nan      |            nan      | nan      |
| opensmile_ridge_homo | artist       |      0 | V     | 1744 | 0.5052 | 0.2064 |    0.8256 | 0.6869 | 0.7126 |   0.672  |   0.9478 |  -0.1559 |              0.0078 |   0.7463 |
| opensmile_ridge_homo | artist       |      0 | A     | 1744 | 0.5082 | 0.2261 |    0.9043 | 0.6899 | 0.7148 |   0.6846 |   0.9444 |  -0.0606 |             -0.0135 |   0.5738 |
| opensmile_svr        | artist       |      0 | V     | 1744 | 0.5177 | 0.2038 |    0.8151 | 0.6902 | 0.72   | nan      | nan      | nan      |            nan      | nan      |
| opensmile_svr        | artist       |      0 | A     | 1744 | 0.5382 | 0.2191 |    0.8763 | 0.7062 | 0.734  | nan      | nan      | nan      |            nan      | nan      |
| opensmile_svr_homo   | artist       |      0 | V     | 1744 | 0.5177 | 0.2038 |    0.8151 | 0.6902 | 0.72   |   0.6577 |   0.9432 |  -0.1676 |              0.0022 |   0.9254 |
| opensmile_svr_homo   | artist       |      0 | A     | 1744 | 0.5382 | 0.2191 |    0.8763 | 0.7062 | 0.734  |   0.672  |   0.9421 |  -0.0922 |             -0.0178 |   0.4583 |

## Paired per-fold differences: opensmile_svr − opensmile_ridge

| split_type   | dim   | metric   |   mean_diff |   sd_diff | folds_B_better   |    t_p |   wilcoxon_p |
|:-------------|:------|:---------|------------:|----------:|:-----------------|-------:|-------------:|
| artist       | A     | R2       |      0.0266 |    0.0465 | 7/10             | 0.1039 |       0.1309 |
| artist       | A     | RMSE     |     -0.0057 |    0.0107 | 7/10             | 0.1246 |       0.1309 |
| artist       | A     | CCC      |      0.0144 |    0.0293 | 7/10             | 0.1537 |       0.1934 |
| artist       | A     | r        |      0.0172 |    0.0292 | 7/10             | 0.0952 |       0.1309 |
| artist       | V     | R2       |      0.0103 |    0.0361 | 5/10             | 0.3894 |       0.4316 |
| artist       | V     | RMSE     |     -0.0022 |    0.0074 | 5/10             | 0.3727 |       0.4316 |
| artist       | V     | CCC      |      0.0033 |    0.0168 | 6/10             | 0.5548 |       0.5566 |
| artist       | V     | r        |      0.0047 |    0.0181 | 5/10             | 0.4347 |       0.5566 |
