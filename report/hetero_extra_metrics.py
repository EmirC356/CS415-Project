"""Supplementary heteroscedastic metrics (seed ensembles, reference tests, oracle control, Spearman).

Run from the repo root:
    python report/hetero_extra_metrics.py
Writes results/metrics_hetero_extra.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PRED = Path("results/predictions")
LABELS = Path("data/interim/labels_fixed.csv")
OUT = Path("results/metrics_hetero_extra.md")
SEEDS = range(5)
BETAS = {"0": "opensmile_hetero_b0", "0.5": "opensmile_hetero_b0.5"}
REFS = {"ridge_homo": "opensmile_ridge_homo__artist__seed0.csv",
        "svr_homo": "opensmile_svr_homo__artist__seed0.csv"}
DIMS = ("V", "A")


def nll(y, mu, sd):
    # same definition as eval/evaluate_v1.py: 0.5 log(2 pi s^2) + (y-mu)^2 / (2 s^2)
    return 0.5 * np.log(2 * np.pi * sd**2) + 0.5 * (y - mu) ** 2 / sd**2


def ensemble(tag: str) -> pd.DataFrame:
    dfs = [pd.read_csv(PRED / f"{tag}__artist__seed{s}.csv") for s in SEEDS]
    assert all(len(d) == 1744 for d in dfs)
    d = pd.concat(dfs, ignore_index=True)
    assert (d.groupby("song_id").fold_id.nunique() == 1).all(), "fold assignment differs across seeds"
    out = d.groupby("song_id").agg(fold_id=("fold_id", "first")).copy()
    for dim in DIMS:
        g = d.groupby("song_id")
        mu = g[f"mu_{dim}"].mean()
        var = (d[f"sigma_{dim}"] ** 2).groupby(d.song_id).mean() + g[f"mu_{dim}"].var(ddof=0)
        out[f"mu_{dim}"], out[f"sigma_{dim}"] = mu, np.sqrt(var)
    return out.reset_index()


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (len(p) - i) * p[idx])
        adj[idx] = min(1.0, run)
    return adj


def per_fold_nll(df, dim, y):
    v = nll(y, df[f"mu_{dim}"].to_numpy(), df[f"sigma_{dim}"].to_numpy())
    return pd.Series(v).groupby(df.fold_id.to_numpy()).mean()


def paired(diff: pd.Series):
    """Paired Wilcoxon summary of diff = learned NLL - other NLL per fold (negative = learned better)."""
    return dict(mean=diff.mean(), better=f"{int((diff < 0).sum())}/{len(diff)}", p=stats.wilcoxon(diff).pvalue)


def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    return float(stats.pearsonr(res(rx), res(ry))[0])


def md(df):
    return df.to_markdown(index=False, floatfmt=".3f")


def main():
    lab = pd.read_csv(LABELS).set_index("song_id")
    ens = {b: ensemble(t) for b, t in BETAS.items()}
    refs = {r: pd.read_csv(PRED / f).merge(pd.DataFrame(index=lab.index).reset_index(), on="song_id") for r, f in REFS.items()}
    for b, e in ens.items():
        ens[b] = e.join(lab[["mu_V", "mu_A", "s_V", "s_A", "s_V_adj", "s_A_adj"]].rename(
            columns={"mu_V": "y_V", "mu_A": "y_A"}), on="song_id")
    for r, d in refs.items():
        refs[r] = d.join(lab[["mu_V", "mu_A"]].rename(columns={"mu_V": "y_V", "mu_A": "y_A"}), on="song_id")

    L = ["# Supplementary heteroscedastic metrics", "",
         "Inputs: `results/predictions/opensmile_hetero_b0__artist__seed{0..4}.csv`, "
         "`opensmile_hetero_b0.5__artist__seed{0..4}.csv`, `opensmile_ridge_homo__artist__seed0.csv`, "
         f"`opensmile_svr_homo__artist__seed0.csv`; labels `{LABELS.as_posix()}` (mu_V, mu_A as targets; s = rater SD, s_adj = adjusted SD).", "",
         "Seed ensemble per beta: mean of member means; variance = mean member variance + variance (ddof=0) of member means. "
         "NLL = 0.5 log(2 pi sigma^2) + (y-mu)^2/(2 sigma^2), averaged per clip within each of the 10 artist-grouped test folds. "
         "Differences are learned-model minus comparison (negative = learned better); 'folds better' counts folds with negative difference; "
         "Wilcoxon signed-rank two-sided, n=10 folds.", ""]

    # per-fold NLL summary
    rows = []
    for b, e in ens.items():
        for d in DIMS:
            f = per_fold_nll(e, d, e[f"y_{d}"].to_numpy())
            rows.append(dict(model=f"hetero beta={b} ensemble", dim=d, mean_NLL=f.mean(), sd_NLL=f.std(ddof=1)))
    for r, df in refs.items():
        for d in DIMS:
            f = per_fold_nll(df, d, df[f"y_{d}"].to_numpy())
            rows.append(dict(model=r, dim=d, mean_NLL=f.mean(), sd_NLL=f.std(ddof=1)))
    L += ["## Per-fold NLL (mean and SD over 10 folds)", "", md(pd.DataFrame(rows)), ""]

    # reference tests (beta=0)
    e0 = ens["0"]
    rows = []
    for r, df in refs.items():
        df = df.set_index("song_id").loc[e0.song_id].reset_index()
        for d in DIMS:
            a = per_fold_nll(e0, d, e0[f"y_{d}"].to_numpy())
            b = per_fold_nll(df, d, df[f"y_{d}"].to_numpy())
            assert (a.index == b.index).all()
            rows.append(dict(reference=r, dim=d, **paired(a - b)))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p.to_numpy())
    L += ["## beta=0 ensemble vs homoscedastic references (per-fold NLL, ensemble minus reference)", "",
          "Holm adjustment over these 4 tests (2 references x 2 dimensions).", "",
          md(t.rename(columns={"mean": "mean_diff", "better": "folds_ens_better", "p": "wilcoxon_p"})), ""]

    # oracle control
    rows = []
    for b, e in ens.items():
        for d in DIMS:
            y, mu = e[f"y_{d}"].to_numpy(), e[f"mu_{d}"].to_numpy()
            rmse = pd.Series((y - mu) ** 2).groupby(e.fold_id.to_numpy()).mean().pow(0.5)
            orc = nll(y, mu, rmse.loc[e.fold_id].to_numpy())
            a = per_fold_nll(e, d, y)
            o = pd.Series(orc).groupby(e.fold_id.to_numpy()).mean()
            rows.append(dict(beta=b, dim=d, learned_NLL=a.mean(), oracle_NLL=o.mean(), **paired(a - o)))
    L += ["## Oracle control: learned sigma vs constant per-fold sigma = fold test RMSE (same ensemble mean)", "",
          "The oracle uses test-fold RMSE, so it is an optimistic constant-sigma control. Difference = learned minus oracle.", "",
          md(pd.DataFrame(rows).rename(columns={"mean": "mean_diff", "better": "folds_learned_better", "p": "wilcoxon_p"})), ""]

    # Spearman
    rows = []
    for b, e in ens.items():
        for d in DIMS:
            y, mu, s = e[f"y_{d}"].to_numpy(), e[f"mu_{d}"].to_numpy(), e[f"sigma_{d}"].to_numpy()
            ae = np.abs(y - mu)
            sr, sa = e[f"s_{d}"].to_numpy(), e[f"s_{d}_adj"].to_numpy()
            sp = lambda x, z: stats.spearmanr(x, z)[0]
            rows.append({"beta": b, "dim": d, "rho(sigma,abs_e)": sp(s, ae),
                         "partial rho(sigma,abs_e given abs_mu)": partial_spearman(s, ae, np.abs(mu)),
                         "rho(sigma,s)": sp(s, sr), "rho(sigma,s_adj)": sp(s, sa),
                         "rho(abs_e,s)": sp(ae, sr), "rho(abs_e,s_adj)": sp(ae, sa)})
    L += ["## Pooled Spearman correlations (all 1,744 clips, out-of-fold)", "",
          "abs_e = |y - mu_hat|, abs_mu = |mu_hat|. Partial rho: ranks of sigma and |e| each residualised (linear) on the rank of |mu_hat|, then Pearson correlation of residuals.", "",
          md(pd.DataFrame(rows)), ""]

    text = "\n".join(L)
    print(text)
    OUT.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
