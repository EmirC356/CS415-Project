"""Shared metrics, version 1 (eval/README.md).

Reads prediction files in the contract format plus the labels, and nothing else.
Per dimension (V, A) and per test fold it computes R² (against the fold's own
mean, sklearn definition), RMSE on the [-1, 1] and the 1-9 scale, CCC and
Pearson r; for sigma models also PICP at 68/95 % and plain Gaussian NLL, and
Spearman rho(sigma, |e|). Fold metrics are summarised as mean ± SD and median.
With --compare A B it reports paired per-fold differences (B − A).

    python eval/evaluate_v1.py results/predictions/opensmile_*__artist__seed0.csv \
        --compare opensmile_ridge opensmile_svr --out results/metrics_baseline.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DIMS = ("V", "A")
Z68, Z95 = stats.norm.ppf(0.84), stats.norm.ppf(0.975)


def ccc(y, p) -> float:
    my, mp = y.mean(), p.mean()
    vy, vp = y.var(), p.var()
    cov = ((y - my) * (p - mp)).mean()
    return float(2 * cov / (vy + vp + (my - mp) ** 2))


def point_metrics(y, p) -> dict:
    e = y - p
    return {
        "R2": float(1 - np.sum(e**2) / np.sum((y - y.mean()) ** 2)),
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "RMSE_19": float(4 * np.sqrt(np.mean(e**2))),
        "CCC": ccc(y, p),
        "r": float(stats.pearsonr(y, p)[0]),
    }


def sigma_metrics(y, p, s) -> dict:
    e = np.abs(y - p)
    return {
        "PICP68": float(np.mean(e <= Z68 * s)),
        "PICP95": float(np.mean(e <= Z95 * s)),
        "NLL": float(np.mean(0.5 * np.log(2 * np.pi * s**2) + 0.5 * (y - p) ** 2 / s**2)),
    }


def load(paths: list[Path], labels: Path) -> pd.DataFrame:
    lab = pd.read_csv(labels)[["song_id", "mu_V", "mu_A", "s_V", "s_A"]]
    lab = lab.rename(columns={"mu_V": "y_V", "mu_A": "y_A"})
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    dup = df.duplicated(["model", "split_type", "seed", "song_id"]).sum()
    assert dup == 0, f"{dup} duplicate predictions"
    return df.merge(lab, on="song_id", how="left", validate="many_to_one")


def per_fold(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, split, seed, k), g in df.groupby(["model", "split_type", "seed", "fold_id"]):
        for d in DIMS:
            y, p = g[f"y_{d}"].to_numpy(), g[f"mu_{d}"].to_numpy()
            m = point_metrics(y, p)
            s = g[f"sigma_{d}"].to_numpy(dtype=float)
            if not np.isnan(s).any():
                m |= sigma_metrics(y, p, s)
            rows.append(dict(model=model, split_type=split, seed=seed, fold=k, dim=d, n=len(g), **m))
    return pd.DataFrame(rows)


def pooled(df: pd.DataFrame) -> pd.DataFrame:
    """Metrics on the pooled out-of-fold predictions (all 1,744 clips)."""
    rows = []
    for (model, split, seed), g in df.groupby(["model", "split_type", "seed"]):
        for d in DIMS:
            y, p = g[f"y_{d}"].to_numpy(), g[f"mu_{d}"].to_numpy()
            m = point_metrics(y, p)
            s = g[f"sigma_{d}"].to_numpy(dtype=float)
            if not np.isnan(s).any():
                m |= sigma_metrics(y, p, s)
                rho, pv = stats.spearmanr(s, np.abs(y - p))
                m |= {"rho_sigma_abs_err": float(rho), "rho_p": float(pv)}
            rows.append(dict(model=model, split_type=split, seed=seed, dim=d, n=len(g), **m))
    return pd.DataFrame(rows)


def summarise(pf: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["R2", "RMSE", "RMSE_19", "CCC", "r", "PICP68", "PICP95", "NLL"] if c in pf]
    g = pf.groupby(["model", "split_type", "dim"])[cols]
    return pd.concat({"mean": g.mean(), "sd": g.std(ddof=1), "median": g.median()}, axis=1)


def paired(pf: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    rows = []
    for (split, d), g in pf.groupby(["split_type", "dim"]):
        ga = g[g.model == a].set_index("fold")
        gb = g[g.model == b].set_index("fold")
        if ga.empty or gb.empty:
            continue
        for met in ["R2", "RMSE", "CCC", "r"]:
            diff = (gb[met] - ga[met]).dropna()
            better = diff > 0 if met != "RMSE" else diff < 0
            t = stats.ttest_1samp(diff, 0.0)
            w = stats.wilcoxon(diff) if len(diff) >= 6 else None
            rows.append(dict(split_type=split, dim=d, metric=met, mean_diff=diff.mean(), sd_diff=diff.std(ddof=1),
                             folds_B_better=f"{int(better.sum())}/{len(diff)}", t_p=t.pvalue,
                             wilcoxon_p=w.pvalue if w else np.nan))
    return pd.DataFrame(rows)


def fmt_summary(s: pd.DataFrame) -> str:
    out = ["| model | split | dim | R² | RMSE | RMSE (1–9) | CCC | Pearson r | PICP68 | PICP95 | NLL |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (model, split, d), r in s.iterrows():
        cell = lambda m: (f"{r[('mean', m)]:.3f} ± {r[('sd', m)]:.3f}"
                          if ("mean", m) in r and not np.isnan(r[("mean", m)]) else "—")
        out.append(f"| {model} | {split} | {d} | " + " | ".join(
            cell(m) for m in ["R2", "RMSE", "RMSE_19", "CCC", "r", "PICP68", "PICP95", "NLL"]) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+", type=Path)
    ap.add_argument("--labels", type=Path, default=Path("data/interim/labels.csv"))
    ap.add_argument("--compare", nargs=2, action="append", metavar=("A", "B"), default=[])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--csv", type=Path, default=None, help="write per-fold metrics here")
    args = ap.parse_args()

    df = load(args.preds, args.labels)
    pf = per_fold(df)
    s = summarise(pf)
    po = pooled(df)
    lines = ["# Metrics (eval/evaluate_v1.py)", "",
             "Inputs: " + ", ".join(f"`{p.as_posix()}`" for p in args.preds), "",
             "## Per-fold mean ± SD (10 test folds)", "", fmt_summary(s), "",
             "## Median over folds", "",
             s.xs("median", axis=1, level=0).round(3).to_markdown(), "",
             "## Pooled out-of-fold (all clips)", "", po.round(4).to_markdown(index=False), ""]
    for a, b in args.compare:
        pr = paired(pf, a, b)
        lines += [f"## Paired per-fold differences: {b} − {a}", "", pr.round(4).to_markdown(index=False), ""]
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    if args.csv:
        pf.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
