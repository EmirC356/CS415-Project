# Supplementary heteroscedastic metrics

Inputs: `results/predictions/opensmile_hetero_b0__artist__seed{0..4}.csv`, `opensmile_hetero_b0.5__artist__seed{0..4}.csv`, `opensmile_ridge_homo__artist__seed0.csv`, `opensmile_svr_homo__artist__seed0.csv`; labels `data/interim/labels_fixed.csv` (mu_V, mu_A as targets; s = rater SD, s_adj = adjusted SD).

Seed ensemble per beta: mean of member means; variance = mean member variance + variance (ddof=0) of member means. NLL = 0.5 log(2 pi sigma^2) + (y-mu)^2/(2 sigma^2), averaged per clip within each of the 10 artist-grouped test folds. Differences are learned-model minus comparison (negative = learned better); 'folds better' counts folds with negative difference; Wilcoxon signed-rank two-sided, n=10 folds.

## Per-fold NLL (mean and SD over 10 folds)

| model                    | dim   |   mean_NLL |   sd_NLL |
|:-------------------------|:------|-----------:|---------:|
| hetero beta=0 ensemble   | V     |     -0.135 |    0.056 |
| hetero beta=0 ensemble   | A     |     -0.267 |    0.046 |
| hetero beta=0.5 ensemble | V     |     -0.115 |    0.065 |
| hetero beta=0.5 ensemble | A     |     -0.249 |    0.064 |
| ridge_homo               | V     |     -0.109 |    0.078 |
| ridge_homo               | A     |     -0.236 |    0.101 |
| svr_homo                 | V     |     -0.116 |    0.058 |
| svr_homo                 | A     |     -0.270 |    0.083 |

## beta=0 ensemble vs homoscedastic references (per-fold NLL, ensemble minus reference)

Holm adjustment over these 4 tests (2 references x 2 dimensions).

| reference   | dim   |   mean_diff | folds_ens_better   |   wilcoxon_p |   p_holm |
|:------------|:------|------------:|:-------------------|-------------:|---------:|
| ridge_homo  | V     |      -0.026 | 8/10               |        0.014 |    0.041 |
| ridge_homo  | A     |      -0.030 | 6/10               |        0.375 |    0.750 |
| svr_homo    | V     |      -0.019 | 9/10               |        0.010 |    0.039 |
| svr_homo    | A     |       0.003 | 3/10               |        0.695 |    0.750 |

## Oracle control: learned sigma vs constant per-fold sigma = fold test RMSE (same ensemble mean)

The oracle uses test-fold RMSE, so it is an optimistic constant-sigma control. Difference = learned minus oracle.

|   beta | dim   |   learned_NLL |   oracle_NLL |   mean_diff | folds_learned_better   |   wilcoxon_p |
|-------:|:------|--------------:|-------------:|------------:|:-----------------------|-------------:|
|  0.000 | V     |        -0.135 |       -0.118 |      -0.017 | 8/10                   |        0.014 |
|  0.000 | A     |        -0.267 |       -0.259 |      -0.008 | 6/10                   |        0.322 |
|  0.500 | V     |        -0.115 |       -0.101 |      -0.014 | 10/10                  |        0.002 |
|  0.500 | A     |        -0.249 |       -0.246 |      -0.003 | 6/10                   |        0.375 |

## Pooled Spearman correlations (all 1,744 clips, out-of-fold)

abs_e = |y - mu_hat|, abs_mu = |mu_hat|. Partial rho: ranks of sigma and |e| each residualised (linear) on the rank of |mu_hat|, then Pearson correlation of residuals.

|   beta | dim   |   rho(sigma,abs_e) |   partial rho(sigma,abs_e given abs_mu) |   rho(sigma,s) |   rho(sigma,s_adj) |   rho(abs_e,s) |   rho(abs_e,s_adj) |
|-------:|:------|-------------------:|----------------------------------------:|---------------:|-------------------:|---------------:|-------------------:|
|  0.000 | V     |              0.167 |                                   0.164 |          0.061 |              0.105 |         -0.123 |              0.015 |
|  0.000 | A     |              0.143 |                                   0.141 |          0.068 |              0.084 |         -0.123 |             -0.038 |
|  0.500 | V     |              0.169 |                                   0.168 |          0.041 |              0.073 |         -0.111 |              0.019 |
|  0.500 | A     |              0.114 |                                   0.113 |          0.039 |              0.063 |         -0.095 |             -0.039 |
