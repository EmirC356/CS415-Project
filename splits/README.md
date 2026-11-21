# splits/ (owner: Emir)

This folder holds the committed fold files. The format and construction rule are in [`docs/contracts.md#fold-file`](../docs/contracts.md#fold-file).

- `folds_artist.csv`: artist-grouped, quadrant-stratified 10-fold, `random_state=415`. Needed before any model run.
- `folds_random.csv`: the same protocol without artist grouping. It measures the artist effect and is made after the progress report.

The fold-generation script and its integrity test live here as well. Nobody trains on a fold file until the integrity test passes.
