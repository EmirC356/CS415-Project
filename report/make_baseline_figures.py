"""Figures and LaTeX tables for the baseline part of the progress report.

    python report/make_baseline_figures.py --raw data/raw

Reads prediction files, labels and fold metrics; writes report/figures/*.pdf
(+ .png previews) and report/tables/*.tex.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from evaluate_v1 import load, per_fold, point_metrics  # noqa: E402

# reference categorical palette (light), slots 1-3; text in neutral ink
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
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
COL_W = 3.45  # IEEE column width (in)


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)


def fig_swap(report_md: Path, out: Path):
    """Static-vs-dynamic correlations per batch (matched vs crossed dimension)."""
    rows = [l for l in report_md.read_text().splitlines() if l.startswith("| 201")]
    vals = {}
    for l in rows:
        c = [x.strip() for x in l.strip("|").split("|")]
        vals[int(c[0])] = dict(matched=(float(c[2]) + float(c[4])) / 2, crossed=(float(c[3]) + float(c[5])) / 2,
                               vv=float(c[2]), va=float(c[3]), aa=float(c[4]), av=float(c[5]))
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    labels = ["static V\n~ dyn. V", "static A\n~ dyn. A", "static V\n~ dyn. A", "static A\n~ dyn. V"]
    keys = ["vv", "aa", "va", "av"]
    x = np.arange(4)
    w = 0.36
    for i, (b, col) in enumerate([(2013, C2), (2014, C1)]):
        v = [vals[b][k] for k in keys]
        bars = ax.bar(x + (i - 0.5) * w, v, w - 0.03, color=col, label=f"{b} ({'744' if b == 2013 else '1000'} clips)")
        for xx, vv in zip(x + (i - 0.5) * w, v):
            ax.text(xx, vv + 0.015, f"{vv:.2f}", ha="center", va="bottom", fontsize=6, color=INK2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Pearson r")
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.axvline(1.5, color=INK2, lw=0.6, ls=":")
    ax.text(0.5, 1.11, "matched dimensions", ha="center", fontsize=7, color=INK2)
    ax.text(2.5, 1.11, "crossed dimensions", ha="center", fontsize=7, color=INK2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
    ax.grid(axis="x", visible=False)
    save(fig, out, "fig_va_swap")


def fig_folds(pf: pd.DataFrame, out: Path):
    """Per-fold R² of ridge vs SVR, artist and random split, per dimension."""
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.9), sharey=True)
    models = [("opensmile_ridge", "Ridge"), ("opensmile_svr", "SVR")]
    for ax, d in zip(axes, ["V", "A"]):
        for j, split in enumerate(["artist", "random"]):
            g = pf[(pf.dim == d) & (pf.split_type == split)]
            for i, (m, lab) in enumerate(models):
                r2 = g[g.model == m].sort_values("fold").R2.to_numpy()
                xpos = j * 2.6 + i
                col = C1 if m.endswith("ridge") else C2
                jitter = np.linspace(-0.18, 0.18, len(r2))
                ax.scatter(np.full_like(r2, xpos) + jitter, r2, s=9, color=col, alpha=0.85, lw=0,
                           label=lab if (j == 0 and d == "V") else None, zorder=3)
                ax.hlines(r2.mean(), xpos - 0.32, xpos + 0.32, color=INK, lw=1.2, zorder=4)
        ax.set_xticks([0.5, 3.1], ["artist-grouped", "random"])
        ax.set_title(DIMNAME[d])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("$R^2$ per test fold")
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=3, frameon=False,
               handles=axes[0].get_legend_handles_labels()[0] +
               [plt.Line2D([], [], color=INK, lw=1.2)],
               labels=["Ridge", "SVR", "mean over folds"])
    fig.subplots_adjust(bottom=0.25)
    save(fig, out, "fig_baseline_folds")


def fig_scatter(df: pd.DataFrame, out: Path, model: str):
    g = df[(df.model == model) & (df.split_type == "artist")]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.85))
    for ax, d in zip(axes, ["V", "A"]):
        y, p = g[f"y_{d}"].to_numpy(), g[f"mu_{d}"].to_numpy()
        m = point_metrics(y, p)
        ax.scatter(y, p, s=3, color=C1, alpha=0.35, lw=0)
        ax.plot([-1, 1], [-1, 1], color=INK2, lw=0.7, ls="--")
        ax.set_xlim(-0.9, 0.9); ax.set_ylim(-0.9, 0.9); ax.set_aspect("equal")
        ax.set_xlabel(f"true {DIMNAME[d].lower()}")
        ax.set_title(f"{DIMNAME[d]}\n$R^2$={m['R2']:.2f}, CCC={m['CCC']:.2f}")
    axes[0].set_ylabel("predicted (out-of-fold)")
    save(fig, out, "fig_baseline_scatter")


def fig_fixes(conds: list[tuple[str, pd.DataFrame]], out: Path):
    """Pooled OOF R² of ridge under the data fixes."""
    fig, ax = plt.subplots(figsize=(COL_W, 1.8))
    x = np.arange(len(conds))
    w = 0.36
    for i, (d, col) in enumerate([("V", C1), ("A", C2)]):
        v = []
        for _, df in conds:
            v.append(point_metrics(df[f"y_{d}"].to_numpy(), df[f"mu_{d}"].to_numpy())["R2"])
        ax.bar(x + (i - 0.5) * w, v, w - 0.03, color=col, label=DIMNAME[d])
        for xx, vv in zip(x + (i - 0.5) * w, v):
            ax.text(xx, vv + 0.01, f"{vv:.2f}", ha="center", va="bottom", fontsize=6, color=INK2)
    ax.set_xticks(x, [c for c, _ in conds])
    ax.set_ylabel("pooled OOF $R^2$ (ridge)")
    ax.set_ylim(0, 0.8)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(axis="x", visible=False)
    save(fig, out, "fig_data_fixes")


def fig_labels_disagreement(df: pd.DataFrame, labels: pd.DataFrame, out: Path):
    """(a) Static labels in the V-A plane; (b) |error| of SVR vs listener sd, by sd decile."""
    lab = labels.set_index("song_id")
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.75), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    ax.scatter(lab.mu_V, lab.mu_A, s=2, color=C1, alpha=0.35, lw=0)
    ax.axhline(0, color=INK2, lw=0.6); ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
    ax.set_xlabel("valence"); ax.set_ylabel("arousal")
    ax.set_title("(a) static labels", fontsize=7)
    ax = axes[1]
    g = df[(df.model == "opensmile_svr") & (df.split_type == "artist")].set_index("song_id")
    for d, col, mk in [("V", C1, "o"), ("A", C2, "s")]:
        e = (g[f"y_{d}"] - g[f"mu_{d}"]).abs()
        s = lab.loc[g.index, f"s_{d}"]
        q = pd.qcut(s, 10, labels=False, duplicates="drop")
        m = e.groupby(q).mean()
        x = s.groupby(q).median()
        ax.plot(4 * x, m, marker=mk, ms=3, lw=1.2, color=col, label=DIMNAME[d])
    ax.set_xlabel("listener sd $s$ (1–9 scale, decile median)")
    ax.set_ylabel("mean $|e|$ (SVR)")
    ax.set_title("(b) error vs. disagreement", fontsize=7)
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout(w_pad=0.8)
    save(fig, out, "fig_labels_disagreement")


def table_main(pf: pd.DataFrame, out: Path):
    order = [("opensmile_ridge", "artist", "Ridge"), ("opensmile_svr", "artist", "SVR"),
             ("opensmile_ridge", "random", "Ridge"), ("opensmile_svr", "random", "SVR")]
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{openSMILE baselines, mean $\pm$ SD over the 10 test folds. RMSE is given on the "
             r"rescaled $[-1,1]$ and on the original 1--9 scale. Best artist-grouped value per column and dimension in bold.}",
             r"\label{tab:baseline}", r"\footnotesize",
             r"\begin{tabular}{@{}lllccccc@{}}", r"\toprule",
             r"Split & Head & Dim & $R^2\uparrow$ & RMSE$\downarrow$ & RMSE$_{1\text{--}9}\downarrow$ & CCC$\uparrow$ & Pearson $r\uparrow$ \\",
             r"\midrule"]
    cols = [("R2", max), ("RMSE", min), ("RMSE_19", min), ("CCC", max), ("r", max)]
    best = {}
    for d in ["V", "A"]:
        g = pf[(pf.split_type == "artist") & (pf.dim == d) & pf.model.isin(["opensmile_ridge", "opensmile_svr"])]
        means = g.groupby("model")[[c for c, _ in cols]].mean()
        best[d] = {c: fn(means[c]) for c, fn in cols}
    for m, split, name in order:
        for d in ["V", "A"]:
            g = pf[(pf.model == m) & (pf.split_type == split) & (pf.dim == d)]
            if g.empty:
                continue

            def f(c):
                s = f"{g[c].mean():.3f}$\\pm${g[c].std(ddof=1):.3f}"
                return f"\\textbf{{{s}}}" if split == "artist" and abs(g[c].mean() - best[d][c]) < 1e-12 else s
            sp = {"artist": "artist-grouped", "random": "random"}[split] if (d == "V" and m.endswith("ridge")) else ""
            lines.append(f"{sp} & {name if d == 'V' else ''} & {d} & " + " & ".join(f(c) for c, _ in cols) + r" \\")
        if split == "artist" and m.endswith("svr"):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "tab_baseline.tex").write_text("\n".join(lines) + "\n")


def table_batch(df: pd.DataFrame, labels: pd.DataFrame, out: Path):
    g = df[(df.model == "opensmile_svr") & (df.split_type == "artist")].merge(
        labels[["song_id", "batch"]], on="song_id")
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Batch & $n$ & $R^2_V$ & $R^2_A$ \\", r"\midrule"]
    for b, gb in g.groupby("batch"):
        rv = point_metrics(gb.y_V.to_numpy(), gb.mu_V.to_numpy())["R2"]
        ra = point_metrics(gb.y_A.to_numpy(), gb.mu_A.to_numpy())["R2"]
        lines.append(f"{int(b)} & {len(gb)} & {rv:.3f} & {ra:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_batch.tex").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path("data/interim/labels_fixed.csv"))
    ap.add_argument("--labels-orig", type=Path, default=Path("data/interim/labels.csv"))
    ap.add_argument("--preds", type=Path, default=Path("results/predictions"))
    ap.add_argument("--out", type=Path, default=Path("report"))
    args = ap.parse_args()
    figs, tabs = args.out / "figures", args.out / "tables"

    files = sorted(args.preds.glob("opensmile_*__seed0.csv"))
    df = load(files, args.labels)
    pf = per_fold(df)
    fig_swap(Path("data/audit/va_swap_report.md"), figs)
    fig_folds(pf, figs)
    fig_scatter(df, figs, "opensmile_svr")
    table_main(pf, tabs)
    table_batch(df, pd.read_csv(args.labels), tabs)
    fig_labels_disagreement(df, pd.read_csv(args.labels), figs)

    ridge = lambda path, lab: load([path], lab).query("split_type == 'artist'")
    conds = [
        ("as supplied", ridge(Path("results/ablation_asis/opensmile_ridge__artist__seed0.csv"), args.labels_orig)),
        ("+ excerpt crop", ridge(Path("results/label_swap_check/opensmile_ridge__artist__seed0.csv"), args.labels_orig)),
        ("+ label fix", ridge(args.preds / "opensmile_ridge__artist__seed0.csv", args.labels)),
    ]
    fig_fixes(conds, figs)
    print("figures:", sorted(p.name for p in figs.glob("*.pdf")))


if __name__ == "__main__":
    main()
