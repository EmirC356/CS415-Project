# Metrics (eval/evaluate_v1.py)

Inputs: `results/predictions/opensmile_ridge__artist__seed0.csv`, `results/predictions/mert_ridge__artist__seed0.csv`, `results/predictions/opensmile_ridge__random__seed0.csv`, `results/predictions/mert_ridge__random__seed0.csv`, `results/predictions/opensmile_svr__artist__seed0.csv`, `results/predictions/mert_ridge_homo__artist__seed0.csv`, `results/predictions/opensmile_ridge_homo__artist__seed0.csv`

## Per-fold mean ± SD (10 test folds)

| model | split | dim | R² | RMSE | RMSE (1–9) | CCC | Pearson r | PICP68 | PICP95 | NLL |
|---|---|---|---|---|---|---|---|---|---|---|
| mert_ridge | artist | A | 0.754 ± 0.026 | 0.164 ± 0.008 | 0.655 ± 0.033 | 0.859 ± 0.019 | 0.871 ± 0.015 | — | — | — |
| mert_ridge | artist | V | 0.563 ± 0.070 | 0.180 ± 0.009 | 0.719 ± 0.038 | 0.726 ± 0.036 | 0.760 ± 0.041 | — | — | — |
| mert_ridge | random | A | 0.765 ± 0.031 | 0.161 ± 0.010 | 0.646 ± 0.040 | 0.866 ± 0.017 | 0.876 ± 0.018 | — | — | — |
| mert_ridge | random | V | 0.590 ± 0.053 | 0.177 ± 0.011 | 0.706 ± 0.045 | 0.743 ± 0.032 | 0.771 ± 0.036 | — | — | — |
| mert_ridge_homo | artist | A | 0.754 ± 0.026 | 0.164 ± 0.008 | 0.655 ± 0.033 | 0.859 ± 0.019 | 0.871 ± 0.015 | 0.662 ± 0.036 | 0.943 ± 0.027 | -0.385 ± 0.058 |
| mert_ridge_homo | artist | V | 0.563 ± 0.070 | 0.180 ± 0.009 | 0.719 ± 0.038 | 0.726 ± 0.036 | 0.760 ± 0.041 | 0.669 ± 0.050 | 0.934 ± 0.034 | -0.288 ± 0.062 |
| opensmile_ridge | artist | A | 0.671 ± 0.061 | 0.189 ± 0.017 | 0.756 ± 0.068 | 0.806 ± 0.041 | 0.822 ± 0.038 | — | — | — |
| opensmile_ridge | artist | V | 0.372 ± 0.079 | 0.216 ± 0.017 | 0.865 ± 0.068 | 0.563 ± 0.070 | 0.622 ± 0.058 | — | — | — |
| opensmile_ridge | random | A | 0.677 ± 0.036 | 0.189 ± 0.010 | 0.757 ± 0.042 | 0.810 ± 0.020 | 0.825 ± 0.022 | — | — | — |
| opensmile_ridge | random | V | 0.395 ± 0.046 | 0.215 ± 0.008 | 0.859 ± 0.034 | 0.579 ± 0.042 | 0.633 ± 0.036 | — | — | — |
| opensmile_ridge_homo | artist | A | 0.671 ± 0.061 | 0.189 ± 0.017 | 0.756 ± 0.068 | 0.806 ± 0.041 | 0.822 ± 0.038 | 0.680 ± 0.058 | 0.949 ± 0.025 | -0.236 ± 0.101 |
| opensmile_ridge_homo | artist | V | 0.372 ± 0.079 | 0.216 ± 0.017 | 0.865 ± 0.068 | 0.563 ± 0.070 | 0.622 ± 0.058 | 0.673 ± 0.051 | 0.949 ± 0.022 | -0.109 ± 0.078 |
| opensmile_svr | artist | A | 0.693 ± 0.045 | 0.183 ± 0.013 | 0.731 ± 0.051 | 0.822 ± 0.029 | 0.835 ± 0.027 | — | — | — |
| opensmile_svr | artist | V | 0.379 ± 0.075 | 0.215 ± 0.012 | 0.859 ± 0.049 | 0.565 ± 0.063 | 0.628 ± 0.062 | — | — | — |

## Median over folds

|                                         |    R2 |   RMSE |   RMSE_19 |   CCC |     r |   PICP68 |   PICP95 |     NLL |
|:----------------------------------------|------:|-------:|----------:|------:|------:|---------:|---------:|--------:|
| ('mert_ridge', 'artist', 'A')           | 0.76  |  0.164 |     0.654 | 0.868 | 0.874 |  nan     |  nan     | nan     |
| ('mert_ridge', 'artist', 'V')           | 0.58  |  0.178 |     0.714 | 0.734 | 0.777 |  nan     |  nan     | nan     |
| ('mert_ridge', 'random', 'A')           | 0.754 |  0.163 |     0.65  | 0.861 | 0.871 |  nan     |  nan     | nan     |
| ('mert_ridge', 'random', 'V')           | 0.592 |  0.176 |     0.706 | 0.732 | 0.775 |  nan     |  nan     | nan     |
| ('mert_ridge_homo', 'artist', 'A')      | 0.76  |  0.164 |     0.654 | 0.868 | 0.874 |    0.668 |    0.951 |  -0.39  |
| ('mert_ridge_homo', 'artist', 'V')      | 0.58  |  0.178 |     0.714 | 0.734 | 0.777 |    0.671 |    0.936 |  -0.298 |
| ('opensmile_ridge', 'artist', 'A')      | 0.678 |  0.184 |     0.735 | 0.811 | 0.825 |  nan     |  nan     | nan     |
| ('opensmile_ridge', 'artist', 'V')      | 0.411 |  0.216 |     0.866 | 0.584 | 0.642 |  nan     |  nan     | nan     |
| ('opensmile_ridge', 'random', 'A')      | 0.683 |  0.187 |     0.748 | 0.81  | 0.828 |  nan     |  nan     | nan     |
| ('opensmile_ridge', 'random', 'V')      | 0.388 |  0.217 |     0.867 | 0.584 | 0.627 |  nan     |  nan     | nan     |
| ('opensmile_ridge_homo', 'artist', 'A') | 0.678 |  0.184 |     0.735 | 0.811 | 0.825 |    0.678 |    0.956 |  -0.264 |
| ('opensmile_ridge_homo', 'artist', 'V') | 0.411 |  0.216 |     0.866 | 0.584 | 0.642 |    0.654 |    0.947 |  -0.103 |
| ('opensmile_svr', 'artist', 'A')        | 0.689 |  0.18  |     0.72  | 0.827 | 0.839 |  nan     |  nan     | nan     |
| ('opensmile_svr', 'artist', 'V')        | 0.405 |  0.215 |     0.859 | 0.587 | 0.645 |  nan     |  nan     | nan     |

## Pooled out-of-fold (all clips)

| model                | split_type   |   seed | dim   |    n |     R2 |   RMSE |   RMSE_19 |    CCC |      r |   PICP68 |   PICP95 |      NLL |   rho_sigma_abs_err |    rho_p |
|:---------------------|:-------------|-------:|:------|-----:|-------:|-------:|----------:|-------:|-------:|---------:|---------:|---------:|--------------------:|---------:|
| mert_ridge           | artist       |      0 | V     | 1744 | 0.573  | 0.1806 |    0.7226 | 0.7288 | 0.757  | nan      | nan      | nan      |            nan      | nan      |
| mert_ridge           | artist       |      0 | A     | 1744 | 0.7577 | 0.1643 |    0.6572 | 0.8617 | 0.8705 | nan      | nan      | nan      |            nan      | nan      |
| mert_ridge           | random       |      0 | V     | 1744 | 0.5908 | 0.1768 |    0.7074 | 0.7426 | 0.7687 | nan      | nan      | nan      |            nan      | nan      |
| mert_ridge           | random       |      0 | A     | 1744 | 0.7653 | 0.1617 |    0.6469 | 0.8667 | 0.8748 | nan      | nan      | nan      |            nan      | nan      |
| mert_ridge_homo      | artist       |      0 | V     | 1744 | 0.573  | 0.1806 |    0.7226 | 0.7288 | 0.757  |   0.6663 |   0.9318 |  -0.2839 |             -0.0354 |   0.1397 |
| mert_ridge_homo      | artist       |      0 | A     | 1744 | 0.7577 | 0.1643 |    0.6572 | 0.8617 | 0.8705 |   0.6606 |   0.9421 |  -0.3821 |             -0.0081 |   0.7349 |
| opensmile_ridge      | artist       |      0 | V     | 1744 | 0.3829 | 0.2172 |    0.8686 | 0.5684 | 0.62   | nan      | nan      | nan      |            nan      | nan      |
| opensmile_ridge      | artist       |      0 | A     | 1744 | 0.6761 | 0.19   |    0.7599 | 0.8105 | 0.8226 | nan      | nan      | nan      |            nan      | nan      |
| opensmile_ridge      | random       |      0 | V     | 1744 | 0.3958 | 0.2149 |    0.8595 | 0.5803 | 0.63   | nan      | nan      | nan      |            nan      | nan      |
| opensmile_ridge      | random       |      0 | A     | 1744 | 0.6774 | 0.1896 |    0.7584 | 0.8104 | 0.8232 | nan      | nan      | nan      |            nan      | nan      |
| opensmile_ridge_homo | artist       |      0 | V     | 1744 | 0.3829 | 0.2172 |    0.8686 | 0.5684 | 0.62   |   0.672  |   0.9478 |  -0.1075 |              0.0129 |   0.5905 |
| opensmile_ridge_homo | artist       |      0 | A     | 1744 | 0.6761 | 0.19   |    0.7599 | 0.8105 | 0.8226 |   0.6829 |   0.9501 |  -0.2341 |             -0.0019 |   0.9355 |
| opensmile_svr        | artist       |      0 | V     | 1744 | 0.3943 | 0.2151 |    0.8606 | 0.5724 | 0.6283 | nan      | nan      | nan      |            nan      | nan      |
| opensmile_svr        | artist       |      0 | A     | 1744 | 0.6987 | 0.1832 |    0.7329 | 0.825  | 0.836  | nan      | nan      | nan      |            nan      | nan      |

## Paired per-fold differences: mert_ridge − opensmile_ridge

| split_type   | dim   | metric   |   mean_diff |   sd_diff | folds_B_better   |    t_p |   wilcoxon_p |
|:-------------|:------|:---------|------------:|----------:|:-----------------|-------:|-------------:|
| artist       | A     | R2       |      0.0829 |    0.0427 | 10/10            | 0.0002 |        0.002 |
| artist       | A     | RMSE     |     -0.0251 |    0.0117 | 10/10            | 0.0001 |        0.002 |
| artist       | A     | CCC      |      0.0528 |    0.025  | 10/10            | 0.0001 |        0.002 |
| artist       | A     | r        |      0.049  |    0.0265 | 10/10            | 0.0002 |        0.002 |
| artist       | V     | R2       |      0.191  |    0.0959 | 10/10            | 0.0001 |        0.002 |
| artist       | V     | RMSE     |     -0.0364 |    0.0175 | 10/10            | 0.0001 |        0.002 |
| artist       | V     | CCC      |      0.1625 |    0.0707 | 10/10            | 0      |        0.002 |
| artist       | V     | r        |      0.1379 |    0.0596 | 10/10            | 0      |        0.002 |
| random       | A     | R2       |      0.0878 |    0.0387 | 10/10            | 0.0001 |        0.002 |
| random       | A     | RMSE     |     -0.0279 |    0.0123 | 10/10            | 0.0001 |        0.002 |
| random       | A     | CCC      |      0.0563 |    0.02   | 10/10            | 0      |        0.002 |
| random       | A     | r        |      0.0506 |    0.0223 | 10/10            | 0.0001 |        0.002 |
| random       | V     | R2       |      0.1954 |    0.0386 | 10/10            | 0      |        0.002 |
| random       | V     | RMSE     |     -0.0382 |    0.0079 | 10/10            | 0      |        0.002 |
| random       | V     | CCC      |      0.1637 |    0.0319 | 10/10            | 0      |        0.002 |
| random       | V     | r        |      0.1388 |    0.0267 | 10/10            | 0      |        0.002 |

## Paired per-fold differences: mert_ridge − opensmile_svr

| split_type   | dim   | metric   |   mean_diff |   sd_diff | folds_B_better   |    t_p |   wilcoxon_p |
|:-------------|:------|:---------|------------:|----------:|:-----------------|-------:|-------------:|
| artist       | A     | R2       |      0.0608 |    0.0312 | 10/10            | 0.0002 |        0.002 |
| artist       | A     | RMSE     |     -0.0189 |    0.0093 | 10/10            | 0.0001 |        0.002 |
| artist       | A     | CCC      |      0.0376 |    0.0144 | 10/10            | 0      |        0.002 |
| artist       | A     | r        |      0.0353 |    0.017  | 10/10            | 0.0001 |        0.002 |
| artist       | V     | R2       |      0.1846 |    0.0777 | 10/10            | 0      |        0.002 |
| artist       | V     | RMSE     |     -0.0351 |    0.0143 | 10/10            | 0      |        0.002 |
| artist       | V     | CCC      |      0.1604 |    0.0648 | 10/10            | 0      |        0.002 |
| artist       | V     | r        |      0.1319 |    0.0604 | 10/10            | 0.0001 |        0.002 |

## Paired per-fold differences: mert_ridge_homo − opensmile_ridge_homo

| split_type   | dim   | metric   |   mean_diff |   sd_diff | folds_B_better   |    t_p |   wilcoxon_p |
|:-------------|:------|:---------|------------:|----------:|:-----------------|-------:|-------------:|
| artist       | A     | R2       |      0.0829 |    0.0427 | 10/10            | 0.0002 |        0.002 |
| artist       | A     | RMSE     |     -0.0251 |    0.0117 | 10/10            | 0.0001 |        0.002 |
| artist       | A     | CCC      |      0.0528 |    0.025  | 10/10            | 0.0001 |        0.002 |
| artist       | A     | r        |      0.049  |    0.0265 | 10/10            | 0.0002 |        0.002 |
| artist       | V     | R2       |      0.191  |    0.0959 | 10/10            | 0.0001 |        0.002 |
| artist       | V     | RMSE     |     -0.0364 |    0.0175 | 10/10            | 0.0001 |        0.002 |
| artist       | V     | CCC      |      0.1625 |    0.0707 | 10/10            | 0      |        0.002 |
| artist       | V     | r        |      0.1379 |    0.0596 | 10/10            | 0      |        0.002 |
