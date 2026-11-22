# Metrics (eval/evaluate_v1.py)

Inputs: `results/predictions/opensmile_ridge__artist__seed0.csv`, `results/predictions/opensmile_ridge__random__seed0.csv`, `results/predictions/opensmile_ridge_homo__artist__seed0.csv`, `results/predictions/opensmile_ridge_homo__random__seed0.csv`, `results/predictions/opensmile_svr__artist__seed0.csv`, `results/predictions/opensmile_svr__random__seed0.csv`, `results/predictions/opensmile_svr_homo__artist__seed0.csv`, `results/predictions/opensmile_svr_homo__random__seed0.csv`

## Per-fold mean ± SD (10 test folds)

| model | split | dim | R² | RMSE | RMSE (1–9) | CCC | Pearson r | PICP68 | PICP95 | NLL |
|---|---|---|---|---|---|---|---|---|---|---|
| opensmile_ridge | artist | A | 0.671 ± 0.061 | 0.189 ± 0.017 | 0.756 ± 0.068 | 0.806 ± 0.041 | 0.822 ± 0.038 | — | — | — |
| opensmile_ridge | artist | V | 0.372 ± 0.079 | 0.216 ± 0.017 | 0.865 ± 0.068 | 0.563 ± 0.070 | 0.622 ± 0.058 | — | — | — |
| opensmile_ridge | random | A | 0.677 ± 0.036 | 0.189 ± 0.010 | 0.757 ± 0.042 | 0.810 ± 0.020 | 0.825 ± 0.022 | — | — | — |
| opensmile_ridge | random | V | 0.395 ± 0.046 | 0.215 ± 0.008 | 0.859 ± 0.034 | 0.579 ± 0.042 | 0.633 ± 0.036 | — | — | — |
| opensmile_ridge_homo | artist | A | 0.671 ± 0.061 | 0.189 ± 0.017 | 0.756 ± 0.068 | 0.806 ± 0.041 | 0.822 ± 0.038 | 0.680 ± 0.058 | 0.949 ± 0.025 | -0.236 ± 0.101 |
| opensmile_ridge_homo | artist | V | 0.372 ± 0.079 | 0.216 ± 0.017 | 0.865 ± 0.068 | 0.563 ± 0.070 | 0.622 ± 0.058 | 0.673 ± 0.051 | 0.949 ± 0.022 | -0.109 ± 0.078 |
| opensmile_ridge_homo | random | A | 0.677 ± 0.036 | 0.189 ± 0.010 | 0.757 ± 0.042 | 0.810 ± 0.020 | 0.825 ± 0.022 | 0.685 ± 0.064 | 0.951 ± 0.021 | -0.237 ± 0.060 |
| opensmile_ridge_homo | random | V | 0.395 ± 0.046 | 0.215 ± 0.008 | 0.859 ± 0.034 | 0.579 ± 0.042 | 0.633 ± 0.036 | 0.677 ± 0.047 | 0.948 ± 0.017 | -0.117 ± 0.039 |
| opensmile_svr | artist | A | 0.693 ± 0.045 | 0.183 ± 0.013 | 0.731 ± 0.051 | 0.822 ± 0.029 | 0.835 ± 0.027 | — | — | — |
| opensmile_svr | artist | V | 0.379 ± 0.075 | 0.215 ± 0.012 | 0.859 ± 0.049 | 0.565 ± 0.063 | 0.628 ± 0.062 | — | — | — |
| opensmile_svr | random | A | 0.698 ± 0.031 | 0.183 ± 0.009 | 0.732 ± 0.034 | 0.823 ± 0.020 | 0.838 ± 0.019 | — | — | — |
| opensmile_svr | random | V | 0.414 ± 0.047 | 0.211 ± 0.009 | 0.845 ± 0.035 | 0.599 ± 0.033 | 0.648 ± 0.033 | — | — | — |
| opensmile_svr_homo | artist | A | 0.693 ± 0.045 | 0.183 ± 0.013 | 0.731 ± 0.051 | 0.822 ± 0.029 | 0.835 ± 0.027 | 0.668 ± 0.061 | 0.943 ± 0.026 | -0.270 ± 0.083 |
| opensmile_svr_homo | artist | V | 0.379 ± 0.075 | 0.215 ± 0.012 | 0.859 ± 0.049 | 0.565 ± 0.063 | 0.628 ± 0.062 | 0.673 ± 0.034 | 0.952 ± 0.018 | -0.116 ± 0.058 |
| opensmile_svr_homo | random | A | 0.698 ± 0.031 | 0.183 ± 0.009 | 0.732 ± 0.034 | 0.823 ± 0.020 | 0.838 ± 0.019 | 0.678 ± 0.037 | 0.950 ± 0.015 | -0.275 ± 0.048 |
| opensmile_svr_homo | random | V | 0.414 ± 0.047 | 0.211 ± 0.009 | 0.845 ± 0.035 | 0.599 ± 0.033 | 0.648 ± 0.033 | 0.675 ± 0.038 | 0.950 ± 0.015 | -0.133 ± 0.041 |

## Median over folds

|                                         |    R2 |   RMSE |   RMSE_19 |   CCC |     r |   PICP68 |   PICP95 |     NLL |
|:----------------------------------------|------:|-------:|----------:|------:|------:|---------:|---------:|--------:|
| ('opensmile_ridge', 'artist', 'A')      | 0.678 |  0.184 |     0.735 | 0.811 | 0.825 |  nan     |  nan     | nan     |
| ('opensmile_ridge', 'artist', 'V')      | 0.411 |  0.216 |     0.866 | 0.584 | 0.642 |  nan     |  nan     | nan     |
| ('opensmile_ridge', 'random', 'A')      | 0.683 |  0.187 |     0.748 | 0.81  | 0.828 |  nan     |  nan     | nan     |
| ('opensmile_ridge', 'random', 'V')      | 0.388 |  0.217 |     0.867 | 0.584 | 0.627 |  nan     |  nan     | nan     |
| ('opensmile_ridge_homo', 'artist', 'A') | 0.678 |  0.184 |     0.735 | 0.811 | 0.825 |    0.678 |    0.956 |  -0.264 |
| ('opensmile_ridge_homo', 'artist', 'V') | 0.411 |  0.216 |     0.866 | 0.584 | 0.642 |    0.654 |    0.947 |  -0.103 |
| ('opensmile_ridge_homo', 'random', 'A') | 0.683 |  0.187 |     0.748 | 0.81  | 0.828 |    0.687 |    0.96  |  -0.257 |
| ('opensmile_ridge_homo', 'random', 'V') | 0.388 |  0.217 |     0.867 | 0.584 | 0.627 |    0.675 |    0.954 |  -0.104 |
| ('opensmile_svr', 'artist', 'A')        | 0.689 |  0.18  |     0.72  | 0.827 | 0.839 |  nan     |  nan     | nan     |
| ('opensmile_svr', 'artist', 'V')        | 0.405 |  0.215 |     0.859 | 0.587 | 0.645 |  nan     |  nan     | nan     |
| ('opensmile_svr', 'random', 'A')        | 0.706 |  0.182 |     0.726 | 0.824 | 0.844 |  nan     |  nan     | nan     |
| ('opensmile_svr', 'random', 'V')        | 0.404 |  0.214 |     0.858 | 0.585 | 0.64  |  nan     |  nan     | nan     |
| ('opensmile_svr_homo', 'artist', 'A')   | 0.689 |  0.18  |     0.72  | 0.827 | 0.839 |    0.689 |    0.943 |  -0.286 |
| ('opensmile_svr_homo', 'artist', 'V')   | 0.405 |  0.215 |     0.859 | 0.587 | 0.645 |    0.673 |    0.948 |  -0.113 |
| ('opensmile_svr_homo', 'random', 'A')   | 0.706 |  0.182 |     0.726 | 0.824 | 0.844 |    0.684 |    0.954 |  -0.287 |
| ('opensmile_svr_homo', 'random', 'V')   | 0.404 |  0.214 |     0.858 | 0.585 | 0.64  |    0.674 |    0.951 |  -0.117 |

## Pooled out-of-fold (all clips)

| model                | split_type   |   seed | dim   |    n |     R2 |   RMSE |   RMSE_19 |    CCC |      r |   PICP68 |   PICP95 |      NLL |   rho_sigma_abs_err |    rho_p |
|:---------------------|:-------------|-------:|:------|-----:|-------:|-------:|----------:|-------:|-------:|---------:|---------:|---------:|--------------------:|---------:|
| opensmile_ridge      | artist       |      0 | V     | 1744 | 0.3829 | 0.2172 |    0.8686 | 0.5684 | 0.62   | nan      | nan      | nan      |            nan      | nan      |
| opensmile_ridge      | artist       |      0 | A     | 1744 | 0.6761 | 0.19   |    0.7599 | 0.8105 | 0.8226 | nan      | nan      | nan      |            nan      | nan      |
| opensmile_ridge      | random       |      0 | V     | 1744 | 0.3958 | 0.2149 |    0.8595 | 0.5803 | 0.63   | nan      | nan      | nan      |            nan      | nan      |
| opensmile_ridge      | random       |      0 | A     | 1744 | 0.6774 | 0.1896 |    0.7584 | 0.8104 | 0.8232 | nan      | nan      | nan      |            nan      | nan      |
| opensmile_ridge_homo | artist       |      0 | V     | 1744 | 0.3829 | 0.2172 |    0.8686 | 0.5684 | 0.62   |   0.672  |   0.9478 |  -0.1075 |              0.0129 |   0.5905 |
| opensmile_ridge_homo | artist       |      0 | A     | 1744 | 0.6761 | 0.19   |    0.7599 | 0.8105 | 0.8226 |   0.6829 |   0.9501 |  -0.2341 |             -0.0019 |   0.9355 |
| opensmile_ridge_homo | random       |      0 | V     | 1744 | 0.3958 | 0.2149 |    0.8595 | 0.5803 | 0.63   |   0.6766 |   0.9484 |  -0.117  |              0.0111 |   0.6433 |
| opensmile_ridge_homo | random       |      0 | A     | 1744 | 0.6774 | 0.1896 |    0.7584 | 0.8104 | 0.8232 |   0.6846 |   0.9513 |  -0.2369 |             -0.0624 |   0.0091 |
| opensmile_svr        | artist       |      0 | V     | 1744 | 0.3943 | 0.2151 |    0.8606 | 0.5724 | 0.6283 | nan      | nan      | nan      |            nan      | nan      |
| opensmile_svr        | artist       |      0 | A     | 1744 | 0.6987 | 0.1832 |    0.7329 | 0.825  | 0.836  | nan      | nan      | nan      |            nan      | nan      |
| opensmile_svr        | random       |      0 | V     | 1744 | 0.4157 | 0.2113 |    0.8453 | 0.5991 | 0.6455 | nan      | nan      | nan      |            nan      | nan      |
| opensmile_svr        | random       |      0 | A     | 1744 | 0.6991 | 0.1831 |    0.7325 | 0.8237 | 0.8361 | nan      | nan      | nan      |            nan      | nan      |
| opensmile_svr_homo   | artist       |      0 | V     | 1744 | 0.3943 | 0.2151 |    0.8606 | 0.5724 | 0.6283 |   0.6726 |   0.9507 |  -0.1162 |              0.0365 |   0.128  |
| opensmile_svr_homo   | artist       |      0 | A     | 1744 | 0.6987 | 0.1832 |    0.7329 | 0.825  | 0.836  |   0.6709 |   0.9432 |  -0.2688 |             -0.0218 |   0.3633 |
| opensmile_svr_homo   | random       |      0 | V     | 1744 | 0.4157 | 0.2113 |    0.8453 | 0.5991 | 0.6455 |   0.6755 |   0.9501 |  -0.1331 |              0.0032 |   0.8948 |
| opensmile_svr_homo   | random       |      0 | A     | 1744 | 0.6991 | 0.1831 |    0.7325 | 0.8237 | 0.8361 |   0.6778 |   0.9501 |  -0.2753 |             -0.0347 |   0.1472 |

## Paired per-fold differences: opensmile_svr − opensmile_ridge

| split_type   | dim   | metric   |   mean_diff |   sd_diff | folds_B_better   |    t_p |   wilcoxon_p |
|:-------------|:------|:---------|------------:|----------:|:-----------------|-------:|-------------:|
| artist       | A     | R2       |      0.0221 |    0.0292 | 8/10             | 0.0405 |       0.0488 |
| artist       | A     | RMSE     |     -0.0062 |    0.008  | 8/10             | 0.0365 |       0.0371 |
| artist       | A     | CCC      |      0.0152 |    0.0177 | 8/10             | 0.0237 |       0.0195 |
| artist       | A     | r        |      0.0137 |    0.0187 | 8/10             | 0.0453 |       0.0645 |
| artist       | V     | R2       |      0.0063 |    0.0448 | 5/10             | 0.6656 |       0.7695 |
| artist       | V     | RMSE     |     -0.0013 |    0.0076 | 5/10             | 0.5954 |       0.6953 |
| artist       | V     | CCC      |      0.0021 |    0.0496 | 6/10             | 0.8962 |       0.8457 |
| artist       | V     | r        |      0.006  |    0.0339 | 5/10             | 0.5895 |       1      |
| random       | A     | R2       |      0.0215 |    0.0324 | 8/10             | 0.0649 |       0.0645 |
| random       | A     | RMSE     |     -0.0064 |    0.0094 | 8/10             | 0.0601 |       0.0645 |
| random       | A     | CCC      |      0.0131 |    0.0197 | 7/10             | 0.0657 |       0.1055 |
| random       | A     | r        |      0.0129 |    0.018  | 8/10             | 0.0503 |       0.0488 |
| random       | V     | R2       |      0.0199 |    0.0166 | 9/10             | 0.0043 |       0.0059 |
| random       | V     | RMSE     |     -0.0036 |    0.003  | 9/10             | 0.0046 |       0.0059 |
| random       | V     | CCC      |      0.0198 |    0.0278 | 7/10             | 0.0507 |       0.0645 |
| random       | V     | r        |      0.0155 |    0.0152 | 9/10             | 0.0103 |       0.0137 |
