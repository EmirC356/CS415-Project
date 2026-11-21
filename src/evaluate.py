"""Evaluation script for 10-fold cross-validation predictions.

Computes:
1. Performance metrics (separately for Valence and Arousal):
   - RMSE
   - R²
   - Lin's Concordance Correlation Coefficient (CCC)
   - Pearson r
   Mean ± std across the 10 folds, presented as a clean Markdown table.
2. Uncertainty metrics (for heteroscedastic model / models with sigma):
   - Gaussian NLL
   - Prediction Interval Coverage Probability (PICP) at 68% and 95%
   - Reliability diagram (calibration curve) saved to outputs/reliability_diagram.png
   - Spearman & Pearson correlation between sigma and actual absolute error |y - mu|
   - Descriptive Spearman correlation between sigma and listener disagreement (valence_std, arousal_std)
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Lin's Concordance Correlation Coefficient (CCC)."""
    if len(y_true) < 2:
        return 0.0
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    if sd_true < 1e-8 or sd_pred < 1e-8:
        return 0.0

    r, _ = stats.pearsonr(y_true, y_pred)
    numerator = 2 * r * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    if denominator < 1e-8:
        return 0.0

    return float(numerator / denominator)


def compute_gaussian_nll(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Compute standard Gaussian Negative Log-Likelihood (per sample)."""
    sigma = np.clip(sigma, 1e-6, None)
    nll = 0.5 * np.log(2 * np.pi * (sigma**2)) + ((y_true - mu) ** 2) / (2 * (sigma**2))
    return float(np.mean(nll))


def compute_picp(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, coverage_nominal: float) -> float:
    """Compute Prediction Interval Coverage Probability (PICP) for a nominal coverage."""
    # z-score for two-tailed normal distribution
    alpha = 1.0 - coverage_nominal
    z = stats.norm.ppf(1.0 - alpha / 2.0)

    lower = mu - z * sigma
    upper = mu + z * sigma
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def plot_reliability_diagram(
    y_v_true: np.ndarray,
    mu_v: np.ndarray,
    sigma_v: np.ndarray,
    y_a_true: np.ndarray,
    mu_a: np.ndarray,
    sigma_a: np.ndarray,
    output_path: str = "outputs/reliability_diagram.png",
):
    """Plot reliability diagram (calibration curves) for Valence and Arousal."""
    nominal_levels = np.linspace(0.05, 0.95, 19)

    picp_v = [compute_picp(y_v_true, mu_v, sigma_v, p) for p in nominal_levels]
    picp_a = [compute_picp(y_a_true, mu_a, sigma_a, p) for p in nominal_levels]

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Ideal Calibration (y = x)", alpha=0.8)
    plt.plot(nominal_levels, picp_v, "o-", color="#1f77b4", label="Valence PICP", linewidth=2, markersize=5)
    plt.plot(nominal_levels, picp_a, "s-", color="#ff7f0e", label="Arousal PICP", linewidth=2, markersize=5)

    # Highlight 68% and 95%
    picp_68_v = compute_picp(y_v_true, mu_v, sigma_v, 0.68)
    picp_95_v = compute_picp(y_v_true, mu_v, sigma_v, 0.95)
    picp_68_a = compute_picp(y_a_true, mu_a, sigma_a, 0.68)
    picp_95_a = compute_picp(y_a_true, mu_a, sigma_a, 0.95)

    plt.scatter([0.68, 0.95], [picp_68_v, picp_95_v], color="#1f77b4", s=80, zorder=5)
    plt.scatter([0.68, 0.95], [picp_68_a, picp_95_a], color="#ff7f0e", s=80, zorder=5)

    plt.xlabel("Nominal Coverage Level", fontsize=12)
    plt.ylabel("Empirical Coverage (PICP)", fontsize=12)
    plt.title("Reliability Diagram (Uncertainty Calibration)", fontsize=13, fontweight="bold")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11, loc="upper left")
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Reliability diagram saved to: %s", output_path)


def evaluate_predictions(
    preds_path: str,
    targets_path: str = "data/processed_targets.csv",
    output_report_path: str | None = None,
    plot_path: str = "outputs/reliability_diagram.png",
) -> str:
    """Run full evaluation on out-of-fold predictions."""
    if not os.path.isfile(preds_path):
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")
    if not os.path.isfile(targets_path):
        raise FileNotFoundError(f"Targets file not found: {targets_path}")

    df_preds = pd.read_csv(preds_path)
    df_targets = pd.read_csv(targets_path)

    # Merge on song_id
    df = df_preds.merge(df_targets, on="song_id", how="inner")
    if len(df) == 0:
        raise ValueError("No common song_ids found between predictions and targets!")

    folds = sorted(df["fold"].unique())
    logger.info("Evaluating %d predictions across %d folds: %s", len(df), len(folds), folds)

    # Fold-level metric collection
    fold_records = []

    for f in folds:
        df_f = df[df["fold"] == f]

        y_v = df_f["valence_mean"].to_numpy()
        y_a = df_f["arousal_mean"].to_numpy()
        mu_v = df_f["mu_v"].to_numpy()
        mu_a = df_f["mu_a"].to_numpy()

        # Valence metrics
        rmse_v = float(np.sqrt(np.mean((y_v - mu_v) ** 2)))
        r2_v = float(1.0 - np.sum((y_v - mu_v) ** 2) / max(1e-8, np.sum((y_v - np.mean(y_v)) ** 2)))
        ccc_v = compute_ccc(y_v, mu_v)
        pearson_v = float(stats.pearsonr(y_v, mu_v)[0]) if len(y_v) > 1 else 0.0

        # Arousal metrics
        rmse_a = float(np.sqrt(np.mean((y_a - mu_a) ** 2)))
        r2_a = float(1.0 - np.sum((y_a - mu_a) ** 2) / max(1e-8, np.sum((y_a - np.mean(y_a)) ** 2)))
        ccc_a = compute_ccc(y_a, mu_a)
        pearson_a = float(stats.pearsonr(y_a, mu_a)[0]) if len(y_a) > 1 else 0.0

        rec = {
            "fold": f,
            "v_rmse": rmse_v,
            "v_r2": r2_v,
            "v_ccc": ccc_v,
            "v_pearson": pearson_v,
            "a_rmse": rmse_a,
            "a_r2": r2_a,
            "a_ccc": ccc_a,
            "a_pearson": pearson_a,
        }

        # Optional sigma metrics per fold
        if "sigma_v" in df_f.columns and "sigma_a" in df_f.columns:
            sig_v = df_f["sigma_v"].to_numpy()
            sig_a = df_f["sigma_a"].to_numpy()
            rec["v_nll"] = compute_gaussian_nll(y_v, mu_v, sig_v)
            rec["a_nll"] = compute_gaussian_nll(y_a, mu_a, sig_a)
            rec["overall_nll"] = 0.5 * (rec["v_nll"] + rec["a_nll"])

        fold_records.append(rec)

    df_fold_metrics = pd.DataFrame(fold_records)

    # Compute overall pooled and fold-averaged metrics
    y_v_all = df["valence_mean"].to_numpy()
    y_a_all = df["arousal_mean"].to_numpy()
    mu_v_all = df["mu_v"].to_numpy()
    mu_a_all = df["mu_a"].to_numpy()

    has_sigma = "sigma_v" in df.columns and "sigma_a" in df.columns
    if has_sigma:
        sig_v_all = df["sigma_v"].to_numpy()
        sig_a_all = df["sigma_a"].to_numpy()
    else:
        sig_v_all = np.full_like(mu_v_all, 0.1)
        sig_a_all = np.full_like(mu_a_all, 0.1)

    # Markdown Table Generation
    lines = []
    lines.append(f"# 10-Fold Cross-Validation Evaluation Report")
    lines.append(f"**Predictions Source**: `{preds_path}`  ")
    lines.append(f"**Total Samples**: {len(df)} | **Number of Folds**: {len(folds)}\n")

    lines.append("## 1. Regression Performance (Valence & Arousal)")
    lines.append(
        "| Fold | Valence RMSE | Valence R² | Valence CCC | Valence Pearson r | Arousal RMSE | Arousal R² | Arousal CCC | Arousal Pearson r |"
    )
    lines.append(
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )

    for _, row in df_fold_metrics.iterrows():
        f = int(row["fold"])
        lines.append(
            f"| Fold {f} | {row['v_rmse']:.4f} | {row['v_r2']:.4f} | {row['v_ccc']:.4f} | {row['v_pearson']:.4f} | "
            f"{row['a_rmse']:.4f} | {row['a_r2']:.4f} | {row['a_ccc']:.4f} | {row['a_pearson']:.4f} |"
        )

    # Mean ± Std Summary Row
    v_rmse_m, v_rmse_s = df_fold_metrics["v_rmse"].mean(), df_fold_metrics["v_rmse"].std()
    v_r2_m, v_r2_s = df_fold_metrics["v_r2"].mean(), df_fold_metrics["v_r2"].std()
    v_ccc_m, v_ccc_s = df_fold_metrics["v_ccc"].mean(), df_fold_metrics["v_ccc"].std()
    v_p_m, v_p_s = df_fold_metrics["v_pearson"].mean(), df_fold_metrics["v_pearson"].std()

    a_rmse_m, a_rmse_s = df_fold_metrics["a_rmse"].mean(), df_fold_metrics["a_rmse"].std()
    a_r2_m, a_r2_s = df_fold_metrics["a_r2"].mean(), df_fold_metrics["a_r2"].std()
    a_ccc_m, a_ccc_s = df_fold_metrics["a_ccc"].mean(), df_fold_metrics["a_ccc"].std()
    a_p_m, a_p_s = df_fold_metrics["a_pearson"].mean(), df_fold_metrics["a_pearson"].std()

    lines.append(
        f"| **Mean ± Std** | **{v_rmse_m:.4f} ± {v_rmse_s:.4f}** | **{v_r2_m:.4f} ± {v_r2_s:.4f}** | "
        f"**{v_ccc_m:.4f} ± {v_ccc_s:.4f}** | **{v_p_m:.4f} ± {v_p_s:.4f}** | "
        f"**{a_rmse_m:.4f} ± {a_rmse_s:.4f}** | **{a_r2_m:.4f} ± {a_r2_s:.4f}** | "
        f"**{a_ccc_m:.4f} ± {a_ccc_s:.4f}** | **{a_p_m:.4f} ± {a_p_s:.4f}** |"
    )

    lines.append("\n## 2. Uncertainty Quantification & Calibration")

    if has_sigma:
        # NLL
        nll_v = compute_gaussian_nll(y_v_all, mu_v_all, sig_v_all)
        nll_a = compute_gaussian_nll(y_a_all, mu_a_all, sig_a_all)
        nll_overall = 0.5 * (nll_v + nll_a)

        # PICP
        picp_68_v = compute_picp(y_v_all, mu_v_all, sig_v_all, 0.68)
        picp_95_v = compute_picp(y_v_all, mu_v_all, sig_v_all, 0.95)
        picp_68_a = compute_picp(y_a_all, mu_a_all, sig_a_all, 0.68)
        picp_95_a = compute_picp(y_a_all, mu_a_all, sig_a_all, 0.95)

        # Error vs Uncertainty Correlation
        abs_err_v = np.abs(y_v_all - mu_v_all)
        abs_err_a = np.abs(y_a_all - mu_a_all)

        err_spearman_v, err_spearman_v_p = stats.spearmanr(sig_v_all, abs_err_v)
        err_pearson_v, err_pearson_v_p = stats.pearsonr(sig_v_all, abs_err_v)

        err_spearman_a, err_spearman_a_p = stats.spearmanr(sig_a_all, abs_err_a)
        err_pearson_a, err_pearson_a_p = stats.pearsonr(sig_a_all, abs_err_a)

        # Disagreement Correlation
        val_std_true = df["valence_std"].to_numpy()
        aro_std_true = df["arousal_std"].to_numpy()

        disagree_v_spearman, disagree_v_p = stats.spearmanr(sig_v_all, val_std_true)
        disagree_a_spearman, disagree_a_p = stats.spearmanr(sig_a_all, aro_std_true)

        lines.append("| Metric | Valence | Arousal | Overall / Average |")
        lines.append("| :--- | :---: | :---: | :---: |")
        lines.append(f"| **Gaussian NLL** | {nll_v:.4f} | {nll_a:.4f} | {nll_overall:.4f} |")
        lines.append(f"| **PICP (68% Nominal)** | {picp_68_v * 100:.2f}% | {picp_68_a * 100:.2f}% | {(picp_68_v + picp_68_a) * 50:.2f}% |")
        lines.append(f"| **PICP (95% Nominal)** | {picp_95_v * 100:.2f}% | {picp_95_a * 100:.2f}% | {(picp_95_v + picp_95_a) * 50:.2f}% |")
        lines.append(f"| **Correlation (sigma vs |y - mu|) Spearman** | {err_spearman_v:.4f} (p={err_spearman_v_p:.2e}) | {err_spearman_a:.4f} (p={err_spearman_a_p:.2e}) | - |")
        lines.append(f"| **Correlation (sigma vs |y - mu|) Pearson** | {err_pearson_v:.4f} (p={err_pearson_v_p:.2e}) | {err_pearson_a:.4f} (p={err_pearson_a_p:.2e}) | - |")
        lines.append(f"| **Descriptive Spearman (sigma vs Listener Disagreement)** | {disagree_v_spearman:.4f} (p={disagree_v_p:.2e}) | {disagree_a_spearman:.4f} (p={disagree_a_p:.2e}) | - |")

        # Generate plot
        plot_reliability_diagram(
            y_v_true=y_v_all,
            mu_v=mu_v_all,
            sigma_v=sig_v_all,
            y_a_true=y_a_all,
            mu_a=mu_a_all,
            sigma_a=sig_a_all,
            output_path=plot_path,
        )
        lines.append(f"\nReliability Diagram plot saved to `{plot_path}`.")
    else:
        lines.append("*(Model does not output predictive standard deviations `sigma_v` and `sigma_a`)*")

    report_str = "\n".join(lines)
    print("\n" + report_str + "\n")

    if output_report_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_report_path)), exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as f:
            f.write(report_str)
        logger.info("Evaluation report saved to: %s", output_report_path)

    return report_str


def main():
    parser = argparse.ArgumentParser(description="Evaluate 10-fold cross-validation predictions.")
    parser.add_argument("--preds", required=True, help="Path to predictions CSV (e.g. outputs/preds_opensmile_heteroscedastic_mlp.csv)")
    parser.add_argument("--targets", default="data/processed_targets.csv", help="Path to processed targets CSV")
    parser.add_argument("--report", default=None, help="Path to save markdown report (default: outputs/eval_<tag>.md)")
    parser.add_argument("--plot", default="outputs/reliability_diagram.png", help="Path to save reliability diagram")
    args = parser.parse_args()

    report_path = args.report
    if report_path is None:
        base_name = os.path.splitext(os.path.basename(args.preds))[0]
        report_path = os.path.join("outputs", f"eval_{base_name}.md")

    evaluate_predictions(
        preds_path=args.preds,
        targets_path=args.targets,
        output_report_path=report_path,
        plot_path=args.plot,
    )


if __name__ == "__main__":
    main()
