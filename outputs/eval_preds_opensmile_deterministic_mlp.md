# 10-Fold Cross-Validation Evaluation Report
**Predictions Source**: `outputs/preds_opensmile_deterministic_mlp.csv`  
**Total Samples**: 1744 | **Number of Folds**: 10

## 1. Regression Performance (Valence & Arousal)
| Fold | Valence RMSE | Valence R² | Valence CCC | Valence Pearson r | Arousal RMSE | Arousal R² | Arousal CCC | Arousal Pearson r |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 0 | 0.2149 | 0.4695 | 0.6851 | 0.7026 | 0.2143 | 0.5608 | 0.7306 | 0.7525 |
| Fold 1 | 0.2246 | 0.4673 | 0.6559 | 0.6859 | 0.2091 | 0.5278 | 0.6970 | 0.7339 |
| Fold 2 | 0.1812 | 0.5640 | 0.7429 | 0.7578 | 0.2045 | 0.5504 | 0.7182 | 0.7425 |
| Fold 3 | 0.2166 | 0.4475 | 0.6787 | 0.6900 | 0.2161 | 0.5406 | 0.7475 | 0.7536 |
| Fold 4 | 0.2107 | 0.4711 | 0.6623 | 0.6894 | 0.2151 | 0.4968 | 0.7140 | 0.7235 |
| Fold 5 | 0.1964 | 0.5768 | 0.7185 | 0.7614 | 0.1969 | 0.6389 | 0.7907 | 0.8015 |
| Fold 6 | 0.1970 | 0.4376 | 0.6192 | 0.6632 | 0.2032 | 0.5188 | 0.6969 | 0.7332 |
| Fold 7 | 0.2173 | 0.4782 | 0.6733 | 0.6969 | 0.2299 | 0.4618 | 0.6902 | 0.6987 |
| Fold 8 | 0.2215 | 0.3933 | 0.6261 | 0.6471 | 0.1908 | 0.6442 | 0.7819 | 0.8027 |
| Fold 9 | 0.2216 | 0.3612 | 0.6210 | 0.6352 | 0.2152 | 0.6070 | 0.7613 | 0.7815 |
| **Mean ± Std** | **0.2102 ± 0.0141** | **0.4666 ± 0.0662** | **0.6683 ± 0.0410** | **0.6930 ± 0.0414** | **0.2095 ± 0.0111** | **0.5547 ± 0.0597** | **0.7328 ± 0.0361** | **0.7524 ± 0.0339** |

## 2. Uncertainty Quantification & Calibration
| Metric | Valence | Arousal | Overall / Average |
| :--- | :---: | :---: | :---: |
| **Gaussian NLL** | -0.0095 | 0.0001 | -0.0047 |
| **PICP (68% Nominal)** | 55.05% | 53.67% | 54.36% |
| **PICP (95% Nominal)** | 84.98% | 84.58% | 84.78% |
| **Correlation (sigma vs |y - mu|) Spearman** | 0.0116 (p=6.27e-01) | 0.0014 (p=9.52e-01) | - |
| **Correlation (sigma vs |y - mu|) Pearson** | 0.0062 (p=7.97e-01) | -0.0062 (p=7.96e-01) | - |
| **Descriptive Spearman (sigma vs Listener Disagreement)** | -0.0389 (p=1.04e-01) | 0.0024 (p=9.20e-01) | - |

Reliability Diagram plot saved to `outputs/reliability_diagram.png`.