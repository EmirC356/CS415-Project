"""Integrity test for splits/folds_artist.csv (docs/contracts.md#fold-file).

Runs with or without labels.csv:
  python splits/test_folds.py            # checks the committed file; adds label checks if
                                         # data/interim/labels.csv exists
  pytest splits/test_folds.py

Must pass before anyone trains on the fold file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ROOT / "splits/folds_artist.csv"
LABELS = ROOT / "data/interim/labels.csv"
N_SPLITS = 10
N_CLIPS = 1744
COLUMNS = ["song_id", "artist_norm", "quadrant", "fold"]


def check(folds: pd.DataFrame, labels: pd.DataFrame | None = None) -> None:
    # Format
    assert list(folds.columns) == COLUMNS, f"columns {list(folds.columns)} != {COLUMNS}"
    assert folds.notna().all().all(), "missing values in fold file"
    assert len(folds) == N_CLIPS, f"{len(folds)} rows, expected {N_CLIPS}"
    assert set(folds["fold"]) == set(range(N_SPLITS)), f"folds {sorted(set(folds['fold']))}"
    assert set(folds["quadrant"]) <= {1, 2, 3, 4}, f"quadrants {sorted(set(folds['quadrant']))}"
    assert (folds["song_id"] < 2001).all(), "2015 clips (id > 2000) present"

    # Every clip exactly once
    assert folds["song_id"].is_unique, "a song_id appears more than once"

    # No artist in two folds (so no artist in train and val/test either)
    span = folds.groupby("artist_norm")["fold"].nunique()
    leaked = span[span > 1]
    assert leaked.empty, f"{len(leaked)} artists span several folds: {leaked.index[:5].tolist()}"

    # Every fold holds every quadrant and is roughly 10 %
    q = pd.crosstab(folds["fold"], folds["quadrant"])
    assert (q > 0).all().all(), "a fold misses a quadrant"
    sizes = q.sum(axis=1)
    assert sizes.min() >= 0.07 * N_CLIPS and sizes.max() <= 0.13 * N_CLIPS, f"fold sizes {sizes.tolist()}"

    if labels is not None:
        lab = labels.set_index(labels["song_id"].astype(int))
        assert set(lab.index) == set(folds["song_id"]), "song ids differ from labels.csv"
        f = folds.set_index("song_id").loc[lab.index]
        assert (f["artist_norm"] == lab["artist_norm"]).all(), "artist_norm differs from labels.csv"
        v_pos, a_pos = lab["mu_V"] >= 0, lab["mu_A"] >= 0
        expected = np.select([v_pos & a_pos, ~v_pos & a_pos, ~v_pos & ~a_pos], [1, 2, 3], default=4)
        assert (f["quadrant"].to_numpy() == expected).all(), "quadrant does not match mu_V, mu_A"


def _load_labels() -> pd.DataFrame | None:
    return pd.read_csv(LABELS) if LABELS.exists() else None


def test_fold_file() -> None:
    check(pd.read_csv(FOLDS), _load_labels())


if __name__ == "__main__":
    labels = _load_labels()
    check(pd.read_csv(FOLDS), labels)
    print(f"[check] {FOLDS.relative_to(ROOT).as_posix()}: PASSED"
          + ("" if labels is not None else " (labels.csv not found: label checks skipped)"))
    sys.exit(0)
