"""Figures, table and numbers for the heteroscedastic-head part of the progress report.

    python report/make_hetero_figures.py --seeds 0 1 2 3 4 --ensemble

Compares the heteroscedastic MLP (beta-NLL, beta = 0 and 0.5) with the constant-sigma
references (`*_homo`, sigma = validation RMSE) on the same artist-grouped folds and the
corrected labels. Metrics come from eval/evaluate_v1.py. Writes

    report/figures/fig_hetero_calibration.{pdf,png}   reliability (coverage) curves, V and A
    report/figures/fig_hetero_sigma.{pdf,png}         mean |error| per sigma decile, V and A
    report/tables/tab_hetero.tex                      calibration / error-ranking table
    results/metrics_hetero.md                         all numbers quoted in the text

With --ensemble the heteroscedastic rows are the seed ensemble: mean of the member
means, and sigma^2 = mean(sigma_i^2) + var(mu_i) (Gaussian-mixture moment matching).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from evaluate_v1 import load, paired, per_fold, pooled  # noqa: E402

# reference categorical palette (light): slot 1 blue, slot 2 orange; the reference is neutral gray
C_B0, C_B05, C_REF = "#2a78d6", "#eb6834", "#7a7975"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIMNAME = {"V": "Valence", "A": "Arousal"}
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})
COL_W = 3.45

REF = "opensmile_ridge_homo"
REF2 = "opensmile_svr_homo"
HET = {"b0": "opensmile_hetero_b0", "b0.5": "opensmile_hetero_b0.5"}
LABEL = {REF: "Ridge, const. $\\sigma$", REF2: "SVR, const. $\\sigma$",
         "b0": "Hetero. MLP, $\\beta{=}0$", "b0.5": "Hetero. MLP, $\\beta{=}0.5$"}


def read(model: str, seed: int) -> Path:
    return Path("results/predictions") / f"{model}__artist__seed{seed}.csv"


def make_ensemble(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Seed ensemble of one model: mean of means, mixture variance."""
    g = df.groupby("song_id")
    out = g[["fold_id", "y_V", "y_A", "s_V", "s_A"]].first()
    for d in ("V", "A"):
        mu = df.pivot(index="song_id", columns="seed", values=f"mu_{d}")
        sg = df.pivot(index="song_id", columns="seed", values=f"sigma_{d}")
        out[f"mu_{d}"] = mu.mean(axis=1)
        out[f"sigma_{d}"] = np.sqrt((sg**2).mean(axis=1) + mu.var(axis=1, ddof=0))
    out = out.reset_index()
    out["model"], out["split_type"], out["seed"] = name, "artist", -1
    return out


def coverage(y, mu, sigma, levels):
    z = stats.norm.ppf(0.5 + levels / 2)
    err = np.abs(y - mu)[:, None]
    return (err <= z[None, :] * sigma[:, None]).mean(axis=0)


def save(fig, name: str):
    out = Path("report/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)


def fig_calibration(df: pd.DataFrame, models: dict[str, str]):
    levels = np.linspace(0.05, 0.95, 19)
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.9), sharey=True)
    colors = {"ref": C_REF, "b0": C_B0, "b0.5": C_B05}
    for ax, d in zip(axes, ("V", "A")):
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":", zorder=1)
        for key, model in models.items():
            g = df[df.model == model]
            cov = coverage(g[f"y_{d}"].to_numpy(), g[f"mu_{d}"].to_numpy(), g[f"sigma_{d}"].to_numpy(dtype=float), levels)
            ax.plot(levels, cov, color=colors[key], lw=1.6, label=LABEL[model if key == "ref" else key], zorder=2)
        ax.set_title(DIMNAME[d])
        ax.set_xlabel("nominal coverage")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    axes[0].set_ylabel("observed coverage")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.5, -0.30))
    save(fig, "fig_hetero_calibration")


def fig_sigma(df: pd.DataFrame, models: dict[str, str]):
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.9), sharey=False)
    colors = {"b0": C_B0, "b0.5": C_B05}
    for ax, d in zip(axes, ("V", "A")):
        ref = df[df.model == models["ref"]]
        ax.axhline(np.abs(ref[f"y_{d}"] - ref[f"mu_{d}"]).mean(), color=C_REF, lw=1.2, ls="--", label=LABEL[models["ref"]])
        for key in ("b0", "b0.5"):
            g = df[df.model == models[key]]
            s = g[f"sigma_{d}"].to_numpy(dtype=float)
            e = np.abs(g[f"y_{d}"] - g[f"mu_{d}"]).to_numpy()
            bins = pd.qcut(s, 10, labels=False, duplicates="drop")
            m = pd.DataFrame({"b": bins, "e": e}).groupby("b").e.mean()
            ax.plot(m.index + 1, m.values, color=colors[key], lw=1.6, marker="o", ms=3.5, mec="white", mew=0.6,
                    label=LABEL[key])
        ax.set_title(DIMNAME[d])
        ax.set_xlabel("predicted $\\sigma$ (decile)")
        ax.set_xticks([1, 3, 5, 7, 9, 10])
    axes[0].set_ylabel("mean $|e|$")
    h, l = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(wspace=0.45)
    fig.legend(h, l, loc="upper center", ncol=1, frameon=False, bbox_to_anchor=(0.5, -0.12))
    save(fig, "fig_hetero_sigma")


def table_numbers(df: pd.DataFrame, labels: pd.DataFrame, order: list[tuple[str, str]]):
    pf = per_fold(df)
    pl = pooled(df).set_index(["model", "dim"])
    lab = labels.set_index("song_id")
    rows, md = [], []
    for key, model in order:
        for d in ("V", "A"):
            f = pf[(pf.model == model) & (pf.dim == d)]
            cell = lambda c: (f"{f[c].mean():.3f}".replace("-", "$-$") + f"$\\pm${f[c].std(ddof=1):.3f}"
                              if c in f and not f[c].isna().all() else "--")
            g = df[df.model == model]
            s = g[f"sigma_{d}"].to_numpy(dtype=float)
            rho = pl.loc[(model, d)].get("rho_sigma_abs_err", np.nan)
            rho_s = stats.spearmanr(s, lab.loc[g.song_id, f"s_{d}"].to_numpy())[0] if not np.isnan(s).any() else np.nan
            rho_sa = stats.spearmanr(s, lab.loc[g.song_id, f"s_{d}_adj"].to_numpy())[0] if not np.isnan(s).any() else np.nan
            fmt = lambda v: "--" if np.isnan(v) else f"{v:.2f}"
            if key == "ref" or model in (REF, REF2):
                rho_cell, rs_cell = "0 (const.)", "0 (const.)"
            else:
                rho_cell, rs_cell = fmt(rho), fmt(rho_s)
            rows.append((LABEL.get(key, LABEL.get(model, model)), d, cell("R2"), cell("PICP68"), cell("PICP95"),
                         cell("NLL"), rho_cell, rs_cell))
            md.append(f"| {model} | {d} | R2 {f['R2'].mean():.3f} | PICP68 {f['PICP68'].mean():.3f} | "
                      f"PICP95 {f['PICP95'].mean():.3f} | NLL {f['NLL'].mean():.3f} | rho(sigma,|e|) {rho:.3f} | "
                      f"rho(sigma,s) {rho_s:.3f} | rho(sigma,s_adj) {rho_sa:.3f} |")
    return rows, md, pf


def write_table(rows, n_seeds: int, ens: bool):
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Heteroscedastic head vs.\ the constant-$\sigma$ reference on openSMILE, artist-grouped folds"
             + (rf" (seed ensemble of {n_seeds})" if ens else "")
             + r". Mean $\pm$ SD over the 10 test folds; $\rho$ is Spearman on all pooled out-of-fold predictions. "
             r"Ideal PICP is $0.68$/$0.95$.}", r"\label{tab:hetero}", r"\footnotesize",
             r"\setlength{\tabcolsep}{2.5pt}",
             r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
             r"Model & D & $R^2\uparrow$ & PICP$_{68}$ & PICP$_{95}$ & NLL$\downarrow$ & $\rho(\sigma,|e|)$ & $\rho(\sigma,s)$ \\",
             r"\midrule"]
    prev = None
    for r in rows:
        if prev is not None and r[0] != prev:
            lines.append(r"\midrule")
        name = r[0] if r[0] != prev else ""
        lines.append(f"{name} & {r[1]} & " + " & ".join(r[2:]) + r" \\")
        prev = r[0]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path("report/tables").mkdir(exist_ok=True)
    Path("report/tables/tab_hetero.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--labels", type=Path, default=Path("data/interim/labels_fixed.csv"))
    args = ap.parse_args()

    labels = pd.read_csv(args.labels)
    paths = [read(REF, 0), read(REF2, 0)] + [read(m, s) for m in HET.values() for s in args.seeds]
    df = load(paths, args.labels)

    keep = [df[df.model.isin([REF, REF2])]]
    models = {"ref": REF}
    for key, m in HET.items():
        g = df[df.model == m]
        if args.ensemble:
            name = f"{m}_ens"
            keep.append(make_ensemble(g, name))
            models[key] = name
        else:
            keep.append(g[g.seed == args.seeds[0]])
            models[key] = m
    df = pd.concat(keep, ignore_index=True)
    LABEL.update({models["b0"]: LABEL["b0"], models["b0.5"]: LABEL["b0.5"]})

    fig_calibration(df, models)
    fig_sigma(df, models)
    order = [("ref", REF), ("ref", REF2), ("b0", models["b0"]), ("b0.5", models["b0.5"])]
    rows, md, pf = table_numbers(df, labels, order)
    write_table(rows, len(args.seeds), args.ensemble)

    text = ["# Heteroscedastic head vs constant-sigma reference (eval/evaluate_v1.py, corrected labels)", "",
            f"seeds: {args.seeds}, ensemble: {args.ensemble}", ""] + md + [""]
    for key in ("b0", "b0.5"):
        text += [f"## paired per-fold: {models[key]} - {REF}", paired(pf, REF, models[key]).round(4).to_markdown(index=False), ""]
    if not args.ensemble and len(args.seeds) > 1:
        # single-seed spread of the same models, to show how big seed noise is
        pfa = per_fold(load([read(m, s) for m in HET.values() for s in args.seeds], args.labels))
        sp = pfa.groupby(["model", "seed", "dim"])[["R2", "NLL", "PICP95"]].mean().groupby(["model", "dim"]).agg(["mean", "std"])
        text += ["## seed spread (mean over folds per seed, then mean/sd over seeds)", sp.round(3).to_markdown(), ""]
    Path("results/metrics_hetero.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text[:14]))


if __name__ == "__main__":
    main()
