# 10-Fold Cross-Validation Evaluation Report
**Predictions Source**: `outputs/preds_opensmile_heteroscedastic_mlp.csv`  
**Total Samples**: 1744 | **Number of Folds**: 10

## 1. Regression Performance (Valence & Arousal)
| Fold | Valence RMSE | Valence R² | Valence CCC | Valence Pearson r | Arousal RMSE | Arousal R² | Arousal CCC | Arousal Pearson r |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 0 | 0.2178 | 0.4554 | 0.6538 | 0.6798 | 0.2204 | 0.5355 | 0.7247 | 0.7490 |
| Fold 1 | 0.2244 | 0.4682 | 0.6694 | 0.6902 | 0.2183 | 0.4851 | 0.6827 | 0.7026 |
| Fold 2 | 0.1837 | 0.5518 | 0.7436 | 0.7529 | 0.2149 | 0.5035 | 0.6951 | 0.7145 |
| Fold 3 | 0.2302 | 0.3759 | 0.6573 | 0.6613 | 0.2260 | 0.4976 | 0.7067 | 0.7261 |
| Fold 4 | 0.2171 | 0.4387 | 0.6247 | 0.6661 | 0.2256 | 0.4463 | 0.6560 | 0.7087 |
| Fold 5 | 0.1971 | 0.5738 | 0.6930 | 0.7725 | 0.2086 | 0.5949 | 0.7525 | 0.7725 |
| Fold 6 | 0.2051 | 0.3904 | 0.6346 | 0.6722 | 0.2117 | 0.4776 | 0.6974 | 0.7363 |
| Fold 7 | 0.2329 | 0.4005 | 0.6030 | 0.6472 | 0.2235 | 0.4912 | 0.6990 | 0.7140 |
| Fold 8 | 0.2164 | 0.4210 | 0.6354 | 0.6672 | 0.1993 | 0.6121 | 0.7621 | 0.7825 |
| Fold 9 | 0.2321 | 0.2992 | 0.5929 | 0.6034 | 0.2292 | 0.5541 | 0.7274 | 0.7482 |
| **Mean ± Std** | **0.2157 ± 0.0161** | **0.4375 ± 0.0814** | **0.6508 ± 0.0442** | **0.6813 ± 0.0491** | **0.2178 ± 0.0093** | **0.5198 ± 0.0533** | **0.7104 ± 0.0320** | **0.7354 ± 0.0273** |

## 2. Uncertainty Quantification & Calibration
| Metric | Valence | Arousal | Overall / Average |
| :--- | :---: | :---: | :---: |
| **Gaussian NLL** | -0.1026 | -0.1094 | -0.1060 |
| **PICP (68% Nominal)** | 65.94% | 65.42% | 65.68% |
| **PICP (95% Nominal)** | 93.81% | 93.41% | 93.61% |
| **Correlation (sigma vs |y - mu|) Spearman** | 0.1019 (p=2.01e-05) | 0.1848 (p=7.37e-15) | - |
| **Correlation (sigma vs |y - mu|) Pearson** | 0.1268 (p=1.08e-07) | 0.2026 (p=1.32e-17) | - |
| **Descriptive Spearman (sigma vs Listener Disagreement)** | 0.0505 (p=3.50e-02) | 0.0542 (p=2.35e-02) | - |

Reliability Diagram plot saved to `outputs/reliability_diagram.png`.