"""openSMILE baseline: ridge and RBF-SVR on the artist-grouped 10-fold split.

For every test fold k, fold (k+1) % 10 is validation and the other 8 folds are
training (docs/contracts.md). Features are standardised with training statistics
only. Hyperparameters are chosen per fold and per dimension on validation RMSE,
then the model is refit on the training folds and scored once on the test fold.
The refit uses the 8 training folds only, so ridge/SVR see exactly the same data
as the MLP heads, which need the validation fold for early stopping.

A homoscedastic reference is written alongside each point model: one constant
sigma per fold and dimension, the RMSE of the validation residuals.

Outputs (prediction-file contract):
    results/predictions/opensmile_<head>__<split>__seed0.csv  (+ .json sidecar)
    results/predictions/opensmile_<head>_homo__<split>__seed0.csv

    python models/baseline_opensmile.py --folds splits/folds_artist.csv --split-type artist
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

DIMS = {"V": "mu_V", "A": "mu_A"}
SEED = 0

RIDGE_GRID = [{"alpha": a} for a in np.logspace(-1, 5, 25)]
SVR_GRID = [
    {"C": c, "gamma_mult": g, "epsilon": e}
    for c, g, e in itertools.product([0.1, 0.3, 1, 3, 10], [0.25, 0.5, 1, 2], [0.02, 0.05, 0.1])
]


def fold_roles(fold: np.ndarray, k: int, n: int = 10) -> np.ndarray:
    role = np.full(fold.shape, "train", dtype=object)
    role[fold == k] = "test"
    role[fold == (k + 1) % n] = "val"
    return role


def make(head: str, p: dict, n_features: int):
    if head == "ridge":
        return Ridge(alpha=p["alpha"])
    # gamma = gamma_mult / n_features is sklearn's 'scale' on standardised inputs
    return SVR(kernel="rbf", C=p["C"], epsilon=p["epsilon"], gamma=p["gamma_mult"] / n_features)


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def run(head: str, X: pd.DataFrame, y: pd.DataFrame, folds: pd.DataFrame, split_type: str, out: Path):
    grid = RIDGE_GRID if head == "ridge" else SVR_GRID
    ids = folds["song_id"].to_numpy()
    Xa = X.loc[ids].to_numpy(dtype=float)
    ya = y.loc[ids]
    fold = folds["fold"].to_numpy()

    rows, homo_rows, chosen = [], [], {}
    for k in range(10):
        role = fold_roles(fold, k)
        tr, va, te = role == "train", role == "val", role == "test"
        sc = StandardScaler().fit(Xa[tr])
        Xtr, Xva, Xte = sc.transform(Xa[tr]), sc.transform(Xa[va]), sc.transform(Xa[te])
        mu, sig = {}, {}
        chosen[k] = {}
        for d, col in DIMS.items():
            ytr, yva = ya[col].to_numpy()[tr], ya[col].to_numpy()[va]
            scores = [(rmse(make(head, p, Xtr.shape[1]).fit(Xtr, ytr).predict(Xva), yva), i)
                      for i, p in enumerate(grid)]
            best_score, best_i = min(scores)
            p = grid[best_i]
            m = make(head, p, Xtr.shape[1]).fit(Xtr, ytr)
            mu[d] = m.predict(Xte)
            sig[d] = rmse(m.predict(Xva), yva)  # homoscedastic sigma from validation residuals
            chosen[k][d] = {**{kk: float(vv) for kk, vv in p.items()}, "val_rmse": best_score,
                            "at_grid_edge": best_i in (0, len(grid) - 1) if head == "ridge" else None}
        base = dict(fold_id=k, split_type=split_type, seed=SEED)
        for j, sid in enumerate(ids[te]):
            rows.append({**base, "model": f"opensmile_{head}", "song_id": int(sid),
                         "mu_V": mu["V"][j], "mu_A": mu["A"][j], "sigma_V": np.nan, "sigma_A": np.nan})
            homo_rows.append({**base, "model": f"opensmile_{head}_homo", "song_id": int(sid),
                              "mu_V": mu["V"][j], "mu_A": mu["A"][j], "sigma_V": sig["V"], "sigma_A": sig["A"]})
        print(f"[{head}/{split_type}] fold {k}: " + ", ".join(
            f"{d} {chosen[k][d]}" for d in DIMS), flush=True)

    out.mkdir(parents=True, exist_ok=True)
    for name, r in [(f"opensmile_{head}", rows), (f"opensmile_{head}_homo", homo_rows)]:
        stem = out / f"{name}__{split_type}__seed{SEED}"
        pd.DataFrame(r).sort_values("song_id").to_csv(stem.with_suffix(".csv"), index=False)
        stem.with_suffix(".json").write_text(json.dumps({
            "model": name, "features": "cache/opensmile/clip_features.parquet (520-d, 45 s excerpt)",
            "head": head, "grid": grid if head == "ridge" else
            {"C": [0.1, 0.3, 1, 3, 10], "gamma": "gamma_mult/n_features, gamma_mult in [0.25,0.5,1,2]",
             "epsilon": [0.02, 0.05, 0.1]},
            "selection": "per fold and dimension, min validation RMSE; refit on the 8 training folds",
            "sigma": "constant per fold and dimension = validation RMSE" if name.endswith("_homo") else None,
            "scaler": "StandardScaler fitted on training folds", "seed": SEED,
            "chosen": chosen,
        }, indent=2, default=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("cache/opensmile/clip_features.parquet"))
    ap.add_argument("--labels", type=Path, default=Path("data/interim/labels.csv"))
    ap.add_argument("--folds", type=Path, default=Path("splits/folds_artist.csv"))
    ap.add_argument("--split-type", default="artist")
    ap.add_argument("--heads", nargs="+", default=["ridge", "svr"])
    ap.add_argument("--out", type=Path, default=Path("results/predictions"))
    args = ap.parse_args()

    X = pd.read_parquet(args.features)
    y = pd.read_csv(args.labels).set_index("song_id")
    folds = pd.read_csv(args.folds).sort_values("song_id").reset_index(drop=True)
    assert set(folds.song_id) <= set(X.index) and set(folds.song_id) <= set(y.index)
    for head in args.heads:
        t = time.time()
        run(head, X, y, folds, args.split_type, args.out)
        print(f"[{head}] done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
