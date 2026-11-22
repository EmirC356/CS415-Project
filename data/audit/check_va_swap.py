"""Check (and optionally fix) a valence/arousal swap in DEAM's 2013 static labels.

Evidence used, per batch (2013 / 2014 from the metadata ``batch`` column):
  1. Pearson r between the static mean ratings and the time-average of the
     per-second dynamic annotations, for matching and crossed dimensions.
  2. Spearman rho between clip loudness (openSMILE RMS energy, mean over the
     45 s excerpt) and each static dimension. Loudness is a well-known arousal cue.

If the crossed correlations exceed the matching ones for a batch, its static
labels have V and A swapped relative to the dynamic annotations. With --fix,
the V and A columns (mu, s, raw, file, adj) of that batch are swapped back and
the result is written to --out with a flag column ``va_swap_fixed``.

    python data/audit/check_va_swap.py --raw data/raw --fix \
        --labels data/interim/labels.csv --out data/interim/labels_fixed.csv \
        --report data/audit/va_swap_report.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = ["mu_{}", "s_{}", "mu_{}_raw", "s_{}_raw", "s_{}_file", "s_{}_adj"]


def dynamic_means(raw: Path) -> pd.DataFrame:
    out = {}
    for d, name in [("V", "valence.csv"), ("A", "arousal.csv")]:
        f = next(p for p in raw.rglob(name) if "averaged per song" in p.as_posix())
        t = pd.read_csv(f)
        t = t.set_index(t.columns[0])
        t.index = t.index.astype(int)
        out[d] = t.mean(axis=1, skipna=True)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--labels", type=Path, default=Path("data/interim/labels.csv"))
    ap.add_argument("--features", type=Path, default=Path("cache/opensmile/clip_features.parquet"))
    ap.add_argument("--out", type=Path, default=Path("data/interim/labels_fixed.csv"))
    ap.add_argument("--report", type=Path, default=Path("data/audit/va_swap_report.md"))
    ap.add_argument("--fix", action="store_true")
    args = ap.parse_args()

    lab = pd.read_csv(args.labels).set_index("song_id")
    # batch from metadata (labels.csv); one 2013 clip has an id >= 1000
    batch = lab["batch"].to_numpy() if lab["batch"].notna().all() else np.where(lab.index < 1000, 2013, 2014)
    dyn = dynamic_means(args.raw).reindex(lab.index)
    energy = None
    if args.features.exists():
        energy = pd.read_parquet(args.features)["pcm_RMSenergy_sma_amean__mean"].reindex(lab.index)

    lines = ["# Valence/arousal swap check (static vs dynamic labels)", "",
             "| batch | n | r(static V, dyn V) | r(static V, dyn A) | r(static A, dyn A) | r(static A, dyn V) |"
             " ρ(energy, static V) | ρ(energy, static A) | verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    swapped = []
    for b in (2013, 2014):
        m = (batch == b) & dyn.notna().all(axis=1).to_numpy()
        g, dd = lab[m], dyn[m]
        r = lambda x, y: np.corrcoef(x, y)[0, 1]
        vv, va, aa, av = r(g.mu_V, dd.V), r(g.mu_V, dd.A), r(g.mu_A, dd.A), r(g.mu_A, dd.V)
        ev = ea = np.nan
        if energy is not None:
            ev = energy[m].corr(g.mu_V, method="spearman")
            ea = energy[m].corr(g.mu_A, method="spearman")
        is_swapped = (va + av) > (vv + aa)
        if is_swapped:
            swapped.append(b)
        lines.append(f"| {b} | {m.sum()} | {vv:.3f} | {va:.3f} | {aa:.3f} | {av:.3f} | {ev:.3f} | {ea:.3f} | "
                     f"{'**V/A swapped**' if is_swapped else 'consistent'} |")

    lines += ["", f"Batches with swapped static labels: {swapped or 'none'}"]
    if args.fix:
        fixed = lab.copy()
        fixed["va_swap_fixed"] = False
        for b in swapped:
            m = batch == b
            for p in PAIRS:
                cv, ca = p.format("V"), p.format("A")
                if cv in fixed and ca in fixed:
                    fixed.loc[m, [cv, ca]] = fixed.loc[m, [ca, cv]].to_numpy()
            fixed.loc[m, "va_swap_fixed"] = True
        fixed.reset_index().to_csv(args.out, index=False)
        lines.append(f"Fixed labels written to `{args.out.as_posix()}` "
                     f"({int(fixed.va_swap_fixed.sum())} clips swapped back).")
    text = "\n".join(lines)
    print(text)
    args.report.write_text(text + "\n")


if __name__ == "__main__":
    main()
