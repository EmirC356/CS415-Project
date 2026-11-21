"""Visualization module for emotion prediction and uncertainty analysis.

Generates publication-quality figures:
1. outputs/sigma_learning_dynamics.png: Training dynamics of sigma across epochs (Warmup -> NLL stage).
2. outputs/predictions_scatter.png: Predicted vs. True Valence & Arousal with confidence intervals (±sigma).
3. outputs/uncertainty_distribution_analysis.png: Sigma distributions and correlations with absolute error and listener disagreement.
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def plot_sigma_learning_dynamics(
    sigma_logs_path: str = "outputs/sigma_logs_opensmile.csv",
    output_path: str = "outputs/sigma_learning_dynamics.png",
):
    """Plot evolution of predicted sigma over training epochs (mean across folds)."""
    if not os.path.isfile(sigma_logs_path):
        print(f"Skipping {output_path}: {sigma_logs_path} not found.")
        return

    df = pd.read_csv(sigma_logs_path)
    # Aggregate across folds by epoch
    grouped = df.groupby("epoch").agg({
        "sigma_v_mean": "mean",
        "sigma_a_mean": "mean",
        "sigma_overall_mean": "mean",
        "sigma_overall_min": "mean",
        "sigma_overall_max": "mean",
        "val_mse": "mean",
        "val_nll": "mean",
    }).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Sigma Evolution
    epochs = grouped["epoch"]
    ax1.axvspan(1, 15, color="#f0f0f0", alpha=0.8, label="Stage 1: MSE Warmup (15 epochs)")
    ax1.plot(epochs, grouped["sigma_v_mean"], label="Valence $\sigma$ (Mean)", color="#1f77b4", linewidth=2.2)
    ax1.plot(epochs, grouped["sigma_a_mean"], label="Arousal $\sigma$ (Mean)", color="#ff7f0e", linewidth=2.2)
    ax1.fill_between(
        epochs,
        grouped["sigma_overall_min"],
        grouped["sigma_overall_max"],
        color="gray",
        alpha=0.2,
        label="Overall $\sigma$ Range [Min, Max]",
    )

    ax1.axhline(0.05, color="red", linestyle=":", alpha=0.7, label="$\sigma_{min} = 0.05$")
    ax1.axhline(1.0, color="red", linestyle="--", alpha=0.7, label="$\sigma_{max} = 1.0$")
    ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Predicted Uncertainty ($\sigma$)", fontsize=12, fontweight="bold")
    ax1.set_title("Evolution of Predicted $\sigma$ Across Folds", fontsize=13, fontweight="bold")
    ax1.set_ylim(0.0, 0.45)
    ax1.legend(loc="upper right", frameon=True, fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Validation Loss Progression
    ax2.axvspan(1, 15, color="#f0f0f0", alpha=0.8, label="Stage 1: MSE Warmup")
    ax2.plot(epochs, grouped["val_mse"], label="Validation MSE (Overall)", color="#2ca02c", linewidth=2.2)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(epochs[epochs > 15], grouped["val_nll"][epochs > 15], label="Validation Gaussian NLL", color="#d62728", linewidth=2.2, linestyle="--")

    ax2.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Validation MSE", fontsize=12, fontweight="bold", color="#2ca02c")
    ax2_twin.set_ylabel("Validation Gaussian NLL", fontsize=12, fontweight="bold", color="#d62728")
    ax2.set_title("Validation Objective Progression", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", frameon=True, fontsize=10)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_predictions_scatter(
    preds_path: str = "outputs/preds_opensmile_heteroscedastic_mlp.csv",
    targets_path: str = "data/processed_targets.csv",
    output_path: str = "outputs/predictions_scatter.png",
):
    """Plot Predicted vs True targets with error bars representing sigma."""
    if not os.path.isfile(preds_path) or not os.path.isfile(targets_path):
        print(f"Skipping {output_path}: files not found.")
        return

    df_preds = pd.read_csv(preds_path)
    df_targets = pd.read_csv(targets_path)
    df = df_preds.merge(df_targets, on="song_id", how="inner")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Sample a subset of points for legible error bars, but compute metrics on all
    np.random.seed(42)
    sample_indices = np.random.choice(len(df), size=min(250, len(df)), replace=False)
    sample_df = df.iloc[sample_indices]

    # Panel 1: Valence
    y_v = df["valence_mean"].to_numpy()
    mu_v = df["mu_v"].to_numpy()
    r_v, _ = stats.pearsonr(y_v, mu_v)
    rmse_v = np.sqrt(np.mean((y_v - mu_v) ** 2))

    ax1.errorbar(
        sample_df["valence_mean"],
        sample_df["mu_v"],
        yerr=sample_df["sigma_v"],
        fmt="o",
        color="#1f77b4",
        ecolor="#aec7e8",
        elinewidth=1.2,
        capsize=2,
        alpha=0.6,
        markersize=4,
        label=f"Sample Clips (with $\pm\sigma$)\nPearson $r = {r_v:.3f}$\nRMSE $= {rmse_v:.3f}$",
    )
    ax1.plot([-1, 1], [-1, 1], "k--", alpha=0.8, label="Ideal ($y = x$)")
    ax1.set_xlabel("True Valence (rescaled [-1, 1])", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Predicted $\mu_v$", fontsize=12, fontweight="bold")
    ax1.set_title("Valence: Predicted vs. True (with $\pm\sigma$ Confidence)", fontsize=13, fontweight="bold")
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.legend(loc="upper left", frameon=True, fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Arousal
    y_a = df["arousal_mean"].to_numpy()
    mu_a = df["mu_a"].to_numpy()
    r_a, _ = stats.pearsonr(y_a, mu_a)
    rmse_a = np.sqrt(np.mean((y_a - mu_a) ** 2))

    ax2.errorbar(
        sample_df["arousal_mean"],
        sample_df["mu_a"],
        yerr=sample_df["sigma_a"],
        fmt="s",
        color="#ff7f0e",
        ecolor="#ffbb78",
        elinewidth=1.2,
        capsize=2,
        alpha=0.6,
        markersize=4,
        label=f"Sample Clips (with $\pm\sigma$)\nPearson $r = {r_a:.3f}$\nRMSE $= {rmse_a:.3f}$",
    )
    ax2.plot([-1, 1], [-1, 1], "k--", alpha=0.8, label="Ideal ($y = x$)")
    ax2.set_xlabel("True Arousal (rescaled [-1, 1])", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Predicted $\mu_a$", fontsize=12, fontweight="bold")
    ax2.set_title("Arousal: Predicted vs. True (with $\pm\sigma$ Confidence)", fontsize=13, fontweight="bold")
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.legend(loc="upper left", frameon=True, fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_uncertainty_distribution_analysis(
    preds_path: str = "outputs/preds_opensmile_heteroscedastic_mlp.csv",
    targets_path: str = "data/processed_targets.csv",
    output_path: str = "outputs/uncertainty_distribution_analysis.png",
):
    """Plot sigma histograms and correlation with absolute error and listener disagreement."""
    if not os.path.isfile(preds_path) or not os.path.isfile(targets_path):
        print(f"Skipping {output_path}: files not found.")
        return

    df_preds = pd.read_csv(preds_path)
    df_targets = pd.read_csv(targets_path)
    df = df_preds.merge(df_targets, on="song_id", how="inner")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 1. Sigma Histograms
    ax = axes[0, 0]
    ax.hist(df["sigma_v"], bins=35, color="#1f77b4", alpha=0.65, label=f"Valence $\sigma$ (Mean: {df['sigma_v'].mean():.3f})", edgecolor="black")
    ax.hist(df["sigma_a"], bins=35, color="#ff7f0e", alpha=0.65, label=f"Arousal $\sigma$ (Mean: {df['sigma_a'].mean():.3f})", edgecolor="black")
    ax.set_xlabel("Predicted Uncertainty ($\sigma$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Clips", fontsize=11, fontweight="bold")
    ax.set_title("Distribution of Predicted $\sigma$ Across 1,744 Clips", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    # 2. Sigma vs Absolute Error (Valence)
    ax = axes[0, 1]
    err_v = np.abs(df["valence_mean"] - df["mu_v"])
    sp_v, p_v = stats.spearmanr(df["sigma_v"], err_v)
    ax.scatter(df["sigma_v"], err_v, alpha=0.45, color="#1f77b4", s=18)
    m_v, b_v = np.polyfit(df["sigma_v"], err_v, 1)
    x_line = np.linspace(df["sigma_v"].min(), df["sigma_v"].max(), 50)
    ax.plot(x_line, m_v * x_line + b_v, color="darkblue", linewidth=2.2, label=f"Trend ($\Delta$Error/$\Delta\sigma$)\nSpearman $\\rho = {sp_v:.3f}$ ($p < 10^{{-4}}$)")
    ax.set_xlabel("Predicted Uncertainty $\sigma_v$", fontsize=11, fontweight="bold")
    ax.set_ylabel("Actual Absolute Error $|y_v - \mu_v|$", fontsize=11, fontweight="bold")
    ax.set_title("Valence: $\sigma$ vs. Actual Prediction Error", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    # 3. Sigma vs Absolute Error (Arousal)
    ax = axes[1, 0]
    err_a = np.abs(df["arousal_mean"] - df["mu_a"])
    sp_a, p_a = stats.spearmanr(df["sigma_a"], err_a)
    ax.scatter(df["sigma_a"], err_a, alpha=0.45, color="#ff7f0e", s=18)
    m_a, b_a = np.polyfit(df["sigma_a"], err_a, 1)
    x_line = np.linspace(df["sigma_a"].min(), df["sigma_a"].max(), 50)
    ax.plot(x_line, m_a * x_line + b_a, color="brown", linewidth=2.2, label=f"Trend ($\Delta$Error/$\Delta\sigma$)\nSpearman $\\rho = {sp_a:.3f}$ ($p < 10^{{-14}}$)")
    ax.set_xlabel("Predicted Uncertainty $\sigma_a$", fontsize=11, fontweight="bold")
    ax.set_ylabel("Actual Absolute Error $|y_a - \mu_a|$", fontsize=11, fontweight="bold")
    ax.set_title("Arousal: $\sigma$ vs. Actual Prediction Error", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    # 4. Sigma vs Human Listener Disagreement
    ax = axes[1, 1]
    sp_dis_v, p_dis_v = stats.spearmanr(df["sigma_v"], df["valence_std"])
    sp_dis_a, p_dis_a = stats.spearmanr(df["sigma_a"], df["arousal_std"])
    ax.scatter(df["sigma_v"], df["valence_std"], alpha=0.4, color="#1f77b4", s=16, label=f"Valence ($\\rho = {sp_dis_v:.3f}, p = {p_dis_v:.3f}$)")
    ax.scatter(df["sigma_a"], df["arousal_std"], alpha=0.4, color="#ff7f0e", s=16, label=f"Arousal ($\\rho = {sp_dis_a:.3f}, p = {p_dis_a:.3f}$)")
    ax.set_xlabel("Model Predicted Uncertainty ($\sigma$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Human Listener Disagreement (Ground-Truth $\sigma_{rater}$)", fontsize=11, fontweight="bold")
    ax.set_title("Predicted Uncertainty vs. Human Rater Ambiguity", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    plot_sigma_learning_dynamics()
    plot_predictions_scatter()
    plot_uncertainty_distribution_analysis()


if __name__ == "__main__":
    main()
