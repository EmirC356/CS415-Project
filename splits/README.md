# splits/ (owner: Emir)

This folder holds the committed fold files. The format and construction rule are in [`docs/contracts.md#fold-file`](../docs/contracts.md#fold-file).

- `folds_artist.csv`: artist-grouped, quadrant-stratified 10-fold, `random_state=415`. Needed before any model run.
- `folds_random.csv`: the same protocol without artist grouping. It measures the artist effect and is made after the progress report.

The fold-generation script and its integrity test live here as well. Nobody trains on a fold file until the integrity test passes.

## Files

| File | What it is |
|---|---|
| `folds_artist.csv` | `song_id, artist_norm, quadrant, fold` for the 1,744 clips |
| `folds_artist_report.md` | Settings, library versions, per-fold sizes and quadrant shares (for the report) |
| `make_folds.py` | Builds both files from `data/interim/labels.csv`, then runs the integrity test |
| `test_folds.py` | Integrity test; runs without `labels.csv` too (then skips the label checks) |

```bash
python splits/make_folds.py     # regenerate (needs data/interim/labels.csv)
python splits/test_folds.py     # check the committed file
```

## Using the folds

For test fold `k`, fold `(k + 1) % 10` is validation and the other 8 folds are training. Use the helper so everyone applies the same rule:

```python
import pandas as pd
from splits.make_folds import fold_roles   # or copy the 5-line function

folds = pd.read_csv("splits/folds_artist.csv")
for k in range(10):
    role = fold_roles(folds["fold"], k)
    train_ids = folds.loc[role == "train", "song_id"]
    val_ids   = folds.loc[role == "val",   "song_id"]
    test_ids  = folds.loc[role == "test",  "song_id"]
```

## Notes

- **Quadrant:** 1 = V≥0, A≥0; 2 = V<0, A≥0; 3 = V<0, A<0; 4 = V≥0, A<0, on the rescaled μ. A μ of exactly 0 counts as positive (52 clips for V, 51 for A).
- **Fold sizes are uneven (134–215 clips).** StratifiedGroupKFold balances quadrant shares, not size, and some artists have up to 23 songs. Report per-fold metrics as mean ± SD as planned; do not re-split.
- **Reproducibility:** the file was made with scikit-learn 1.7.2 (see the report). Another sklearn version may shuffle differently, so the committed CSV is the reference; do not regenerate it locally and train on your own copy.
- **Grouping key check:** `artist_norm` gives all 67 artists that appear in both 2013 and 2014 a single key. Every group of the old prototype key (root `splits.csv`) lies within one `artist_norm` group.
- **Legacy file:** the root `splits.csv` is the pre-contract prototype split (GroupKFold, no stratification, seed 42). Do not use it for report results.
