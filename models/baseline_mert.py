"""Frozen MERT-v1-95M + ridge, same protocol as models/baseline_opensmile.py.

Input: cache/mert95m/<song_id>.npy, shape (windows, 13, 768). Each clip is
mean-pooled over its 5 s windows, giving 13 x 768. For every test fold k,
fold (k+1) % 10 is validation and the other 8 folds are training
(docs/contracts.md). Features are standardised with training statistics only.
The layer AND ridge alpha are chosen jointly per fold and per dimension on
validation RMSE; the model is then refit on the 8 training folds and scored
once on the test fold. The alpha grid and the refit rule match the openSMILE
ridge, so the comparison differs only in the representation (same head).

A homoscedastic reference is written alongside: one constant sigma per fold
and dimension, the RMSE of the validation residuals.

Outputs (prediction-file contract):
    results/predictions/mert_ridge__<split>__seed0.csv  (+ .json sidecar)
    results/predictions/mert_ridge_homo__<split>__seed0.csv

    python models/baseline_mert.py --labels data/interim/labels_fixed.csv \
        --folds splits/folds_artist.csv --split-type artist
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DIMS = {"V": "mu_V", "A": "mu_A"}
SEED = 0
ALPHAS = np.logspace(-1, 5, 25)          # same grid as the openSMILE ridge
N_LAYERS = 13


def fold_roles(fold: np.ndarray, k: int, n: int = 10) -> np.ndarray:
    role = np.full(fold.shape, "train", dtype=object)
    role[fold == k] = "test"
    role[fold == (k + 1) % n] = "val"
    return role


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def load_mert(cache: Path, ids: np.ndarray) -> np.ndarray:
    """(N, 13, 768) float32, window-mean pooled."""
    return np.stack([np.load(cache / f"{i}.npy").astype(np.float32).mean(axis=0) for i in ids])


def run(E: np.ndarray, y: pd.DataFrame, folds: pd.DataFrame, split_type: str, out: Path) -> None:
    ids = folds["song_id"].to_numpy()
    fold = folds["fold"].to_numpy()
    rows, homo_rows, chosen = [], [], {}
    for k in range(10):
        role = fold_roles(fold, k)
        tr, va, te = role == "train", role == "val", role == "test"
        mu, sig = {}, {}
        chosen[k] = {}
        best = {d: (np.inf, None, None) for d in DIMS}
        scalers = {}
        for L in range(N_LAYERS):
            sc = StandardScaler().fit(E[tr, L])
            scalers[L] = sc
            Xtr, Xva = sc.transform(E[tr, L]), sc.transform(E[va, L])
            for d, col in DIMS.items():
                ytr, yva = y[col].to_numpy()[tr], y[col].to_numpy()[va]
                for i, a in enumerate(ALPHAS):
                    s = rmse(Ridge(alpha=a).fit(Xtr, ytr).predict(Xva), yva)
                    if s < best[d][0]:
                        best[d] = (s, L, i)
        for d, col in DIMS.items():
            s, L, i = best[d]
            sc = scalers[L]
            m = Ridge(alpha=ALPHAS[i]).fit(sc.transform(E[tr, L]), y[col].to_numpy()[tr])
            mu[d] = m.predict(sc.transform(E[te, L]))
            sig[d] = rmse(m.predict(sc.transform(E[va, L])), y[col].to_numpy()[va])
            chosen[k][d] = {"layer": int(L), "alpha": float(ALPHAS[i]), "val_rmse": s,
                            "at_alpha_edge": i in (0, len(ALPHAS) - 1)}
        base = dict(fold_id=k, split_type=split_type, seed=SEED)
        for j, sid in enumerate(ids[te]):
            rows.append({**base, "model": "mert_ridge", "song_id": int(sid),
                         "mu_V": mu["V"][j], "mu_A": mu["A"][j], "sigma_V": np.nan, "sigma_A": np.nan})
            homo_rows.append({**base, "model": "mert_ridge_homo", "song_id": int(sid),
                              "mu_V": mu["V"][j], "mu_A": mu["A"][j],
                              "sigma_V": sig["V"], "sigma_A": sig["A"]})
        print(f"[mert_ridge/{split_type}] fold {k}: " + ", ".join(
            f"{d} L{chosen[k][d]['layer']} a={chosen[k][d]['alpha']:.3g}" for d in DIMS), flush=True)

    out.mkdir(parents=True, exist_ok=True)
    for name, r in [("mert_ridge", rows), ("mert_ridge_homo", homo_rows)]:
        stem = out / f"{name}__{split_type}__seed{SEED}"
        pd.DataFrame(r).sort_values("song_id").to_csv(stem.with_suffix(".csv"), index=False)
        stem.with_suffix(".json").write_text(json.dumps({
            "model": name,
            "features": "cache/mert95m (MERT-v1-95M, 24 kHz, 5 s windows, window-mean, 13 layers)",
            "head": "ridge", "alphas": ALPHAS.tolist(),
            "selection": "per fold and dimension, layer x alpha by min validation RMSE; "
                         "refit on the 8 training folds",
            "sigma": "constant per fold and dimension = validation RMSE" if name.endswith("_homo") else None,
            "scaler": "StandardScaler fitted on training folds, per layer", "seed": SEED,
            "chosen": chosen,
        }, indent=2, default=float), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("cache/mert95m"))
    ap.add_argument("--labels", type=Path, default=Path("data/interim/labels_fixed.csv"))
    ap.add_argument("--folds", type=Path, default=Path("splits/folds_artist.csv"))
    ap.add_argument("--split-type", default="artist")
    ap.add_argument("--out", type=Path, default=Path("results/predictions"))
    args = ap.parse_args()

    y = pd.read_csv(args.labels).set_index("song_id")
    folds = pd.read_csv(args.folds).sort_values("song_id").reset_index(drop=True)
    y = y.loc[folds["song_id"]]
    t = time.time()
    E = load_mert(args.cache, folds["song_id"].to_numpy())
    print(f"loaded {E.shape} in {time.time() - t:.0f}s")
    run(E, y, folds, args.split_type, args.out)
    print(f"done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
