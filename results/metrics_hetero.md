# Heteroscedastic head vs constant-sigma reference (eval/evaluate_v1.py, corrected labels)

seeds: [0, 1, 2, 3, 4], ensemble: True

| opensmile_ridge_homo | V | R2 0.372 | PICP68 0.673 | PICP95 0.949 | NLL -0.109 | rho(sigma,|e|) 0.013 | rho(sigma,s) -0.013 | rho(sigma,s_adj) 0.012 |
| opensmile_ridge_homo | A | R2 0.671 | PICP68 0.680 | PICP95 0.949 | NLL -0.236 | rho(sigma,|e|) -0.002 | rho(sigma,s) -0.005 | rho(sigma,s_adj) -0.025 |
| opensmile_svr_homo | V | R2 0.379 | PICP68 0.673 | PICP95 0.952 | NLL -0.116 | rho(sigma,|e|) 0.036 | rho(sigma,s) -0.025 | rho(sigma,s_adj) -0.013 |
| opensmile_svr_homo | A | R2 0.693 | PICP68 0.668 | PICP95 0.943 | NLL -0.270 | rho(sigma,|e|) -0.022 | rho(sigma,s) 0.006 | rho(sigma,s_adj) -0.029 |
| opensmile_hetero_b0_ens | V | R2 0.375 | PICP68 0.664 | PICP95 0.948 | NLL -0.135 | rho(sigma,|e|) 0.167 | rho(sigma,s) 0.061 | rho(sigma,s_adj) 0.065 |
| opensmile_hetero_b0_ens | A | R2 0.679 | PICP68 0.700 | PICP95 0.962 | NLL -0.267 | rho(sigma,|e|) 0.143 | rho(sigma,s) 0.068 | rho(sigma,s_adj) 0.120 |
| opensmile_hetero_b0.5_ens | V | R2 0.353 | PICP68 0.656 | PICP95 0.939 | NLL -0.115 | rho(sigma,|e|) 0.169 | rho(sigma,s) 0.041 | rho(sigma,s_adj) 0.038 |
| opensmile_hetero_b0.5_ens | A | R2 0.671 | PICP68 0.682 | PICP95 0.949 | NLL -0.249 | rho(sigma,|e|) 0.114 | rho(sigma,s) 0.040 | rho(sigma,s_adj) 0.097 |

## paired per-fold: opensmile_hetero_b0_ens - opensmile_ridge_homo
| split_type   | dim   | metric   |   mean_diff |   sd_diff | folds_B_better   |    t_p |   wilcoxon_p |
|:-------------|:------|:---------|------------:|----------:|:-----------------|-------:|-------------:|
| artist       | A     | R2       |      0.0077 |    0.0416 | 6/10             | 0.5709 |       0.625  |
| artist       | A     | RMSE     |     -0.0018 |    0.0113 | 6/10             | 0.6186 |       0.625  |
| artist       | A     | CCC      |      0.0066 |    0.027  | 5/10             | 0.4596 |       1      |
| artist       | A     | r        |      0.0048 |    0.0262 | 5/10             | 0.5727 |       0.7695 |
| artist       | V     | R2       |      0.0026 |    0.05   | 5/10             | 0.8716 |       0.9219 |
| artist       | V     | RMSE     |     -0.0007 |    0.0082 | 5/10             | 0.8032 |       0.9219 |
| artist       | V     | CCC      |      0.0146 |    0.0585 | 4/10             | 0.4499 |       0.9219 |
| artist       | V     | r        |      0.0024 |    0.0324 | 4/10             | 0.8189 |       0.7695 |

## paired per-fold: opensmile_hetero_b0.5_ens - opensmile_ridge_homo
| split_type   | dim   | metric   |   mean_diff |   sd_diff | folds_B_better   |    t_p |   wilcoxon_p |
|:-------------|:------|:---------|------------:|----------:|:-----------------|-------:|-------------:|
| artist       | A     | R2       |     -0.0007 |    0.0514 | 5/10             | 0.9658 |       1      |
| artist       | A     | RMSE     |      0.0006 |    0.0144 | 5/10             | 0.8943 |       0.9219 |
| artist       | A     | CCC      |      0.007  |    0.0334 | 5/10             | 0.5223 |       0.6953 |
| artist       | A     | r        |      0.0015 |    0.0308 | 5/10             | 0.8827 |       0.9219 |
| artist       | V     | R2       |     -0.0188 |    0.0617 | 3/10             | 0.3606 |       0.3223 |
| artist       | V     | RMSE     |      0.0029 |    0.0098 | 3/10             | 0.3749 |       0.3223 |
| artist       | V     | CCC      |      0.0226 |    0.0623 | 6/10             | 0.2814 |       0.2754 |
| artist       | V     | r        |     -0.0048 |    0.0361 | 3/10             | 0.686  |       0.4316 |
