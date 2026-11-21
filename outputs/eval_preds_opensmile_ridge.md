# 10-Fold Cross-Validation Evaluation Report
**Predictions Source**: `outputs/preds_opensmile_ridge.csv`  
**Total Samples**: 1744 | **Number of Folds**: 10

## 1. Regression Performance (Valence & Arousal)
| Fold | Valence RMSE | Valence R² | Valence CCC | Valence Pearson r | Arousal RMSE | Arousal R² | Arousal CCC | Arousal Pearson r |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 0 | 0.2204 | 0.4421 | 0.6121 | 0.6676 | 0.2365 | 0.4652 | 0.6638 | 0.7048 |
| Fold 1 | 0.2150 | 0.5121 | 0.6749 | 0.7178 | 0.2346 | 0.4053 | 0.6168 | 0.6485 |
| Fold 2 | 0.1867 | 0.5371 | 0.7056 | 0.7468 | 0.2128 | 0.5131 | 0.6863 | 0.7205 |
| Fold 3 | 0.2173 | 0.4443 | 0.6742 | 0.6915 | 0.2390 | 0.4380 | 0.6744 | 0.7051 |
| Fold 4 | 0.2073 | 0.4880 | 0.6483 | 0.7047 | 0.2161 | 0.4920 | 0.6692 | 0.7048 |
| Fold 5 | 0.1918 | 0.5962 | 0.7355 | 0.7741 | 0.2088 | 0.5942 | 0.7458 | 0.7767 |
| Fold 6 | 0.1936 | 0.4566 | 0.6473 | 0.6813 | 0.2283 | 0.3927 | 0.6299 | 0.6561 |
| Fold 7 | 0.2194 | 0.4680 | 0.6452 | 0.6897 | 0.2154 | 0.5272 | 0.7092 | 0.7297 |
| Fold 8 | 0.2169 | 0.4179 | 0.6358 | 0.6708 | 0.2164 | 0.5426 | 0.7241 | 0.7401 |
| Fold 9 | 0.2244 | 0.3446 | 0.5830 | 0.6341 | 0.2102 | 0.6252 | 0.7461 | 0.7983 |
| **Mean ± Std** | **0.2093 ± 0.0136** | **0.4707 ± 0.0688** | **0.6562 ± 0.0440** | **0.6978 ± 0.0405** | **0.2218 ± 0.0116** | **0.4996 ± 0.0767** | **0.6866 ± 0.0448** | **0.7184 ± 0.0468** |

## 2. Uncertainty Quantification & Calibration
| Metric | Valence | Arousal | Overall / Average |
| :--- | :---: | :---: | :---: |
| **Gaussian NLL** | -0.1283 | -0.0628 | -0.0955 |
| **PICP (68% Nominal)** | 62.21% | 61.81% | 62.01% |
| **PICP (95% Nominal)** | 92.43% | 91.11% | 91.77% |
| **Correlation (sigma vs |y - mu|) Spearman** | -0.0442 (p=6.51e-02) | -0.0001 (p=9.96e-01) | - |
| **Correlation (sigma vs |y - mu|) Pearson** | -0.0558 (p=1.99e-02) | -0.0145 (p=5.46e-01) | - |
| **Descriptive Spearman (sigma vs Listener Disagreement)** | 0.0089 (p=7.12e-01) | 0.0153 (p=5.23e-01) | - |

Reliability Diagram plot saved to `outputs/reliability_diagram.png`.