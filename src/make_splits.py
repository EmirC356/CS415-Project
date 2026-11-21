"""Build cleaned DEAM targets and an artist-disjoint 10-fold split.

Outputs
-------
data/processed_targets.csv : song_id, valence_mean, arousal_mean, valence_std, arousal_std
                             (valence/arousal means rescaled [1,9] -> [-1,1], stds /4)
splits.csv                 : song_id, artist, fold  (10-fold GroupKFold on normalized artist)

Run:  python src/make_splits.py
"""

from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ANNOT = (
    DATA
    / "DEAM_Annotations"
    / "annotations"
    / "annotations averaged per song"
    / "song_level"
    / "static_annotations_averaged_songs_1_2000.csv"
)
META = {
    "2013": DATA / "metadata" / "metadata_2013.csv",
    "2014": DATA / "metadata" / "metadata_2014.csv",
    "2015": DATA / "metadata" / "metadata_2015.csv",
}

TARGET_COLS = ["song_id", "valence_mean", "arousal_mean", "valence_std", "arousal_std"]
N_SPLITS = 10
SEED = 42

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# 1. Targets
# --------------------------------------------------------------------------- #
def load_targets() -> pd.DataFrame:
    """Load the 1,744 clip targets and rescale to [-1, 1]."""
    raw = pd.read_csv(ANNOT)
    # Headers ship with stray leading spaces (" valence_mean").
    raw.columns = [c.strip() for c in raw.columns]

    df = raw[TARGET_COLS].copy()
    df["song_id"] = df["song_id"].astype(int)

    df["valence_mean"] = (df["valence_mean"] - 5.0) / 4.0
    df["arousal_mean"] = (df["arousal_mean"] - 5.0) / 4.0
    df["valence_std"] = df["valence_std"] / 4.0
    df["arousal_std"] = df["arousal_std"] / 4.0

    assert df["song_id"].is_unique, "duplicate song_id in targets"
    assert df[["valence_mean", "arousal_mean"]].abs().max().max() <= 1.0 + 1e-9, (
        "rescaled means outside [-1, 1]"
    )
    return df.sort_values("song_id").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Artist metadata
# --------------------------------------------------------------------------- #
def normalize_artist(name: object) -> str:
    """Lowercase, fold accents, strip punctuation and collapse whitespace."""
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.casefold()
    s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def _read_2013(path: Path) -> pd.DataFrame:
    """song_id, file_name, Artist, ... (values carry tabs/quotes)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame(
        {
            "song_id": pd.to_numeric(df["song_id"], errors="coerce"),
            "artist_raw": df["Artist"].astype(str).str.strip(),
        }
    )
    return out.dropna(subset=["song_id"]).assign(song_id=lambda d: d.song_id.astype(int))


def _read_2014(path: Path) -> pd.DataFrame:
    """Id,Artist,Album,Track,Genre,segment start,segment end,last.fm labels...

    The trailing last.fm labels column is an unquoted, comma-ridden free-text
    list with ragged lengths, so pandas' parser fails. We only need the first
    eight fields: split on the first seven commas, then require that the id and
    both segment offsets parse as numbers. Rows that fail are ragged
    continuation lines from the previous record and get dropped.
    """
    rows: list[tuple[int, str]] = []
    dropped = 0
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as fh:
        next(fh)  # header
        for line in fh:
            parts = line.rstrip("\r\n").split(",", 7)
            if len(parts) < 7:
                dropped += 1
                continue
            try:
                song_id = int(float(parts[0]))
                float(parts[5])  # segment start
                float(parts[6].split(",")[0])  # segment end (labels follow it)
            except ValueError:
                dropped += 1
                continue
            rows.append((song_id, parts[1].strip()))
    print(f"[meta] 2014: kept {len(rows)} rows, dropped {dropped} ragged lines")
    return pd.DataFrame(rows, columns=["song_id", "artist_raw"])


def _read_2015(path: Path) -> pd.DataFrame:
    """id,Filename,title,artist,album,genre,... (ids 2001+, outside the 1,744).

    Same free-text tail problem as 2014, and some titles carry unquoted commas
    that shift the columns. Tolerant split on the first five commas; rows whose
    id is not an integer are dropped. These ids never join to the 1,744 clips.
    """
    rows: list[tuple[int, str]] = []
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as fh:
        next(fh)
        for line in fh:
            parts = line.rstrip("\r\n").split(",", 5)
            if len(parts) < 4:
                continue
            try:
                song_id = int(float(parts[0]))
            except ValueError:
                continue
            rows.append((song_id, parts[3].strip()))
    return pd.DataFrame(rows, columns=["song_id", "artist_raw"])


def load_artist_map() -> tuple[pd.DataFrame, dict[str, int]]:
    """Merge every metadata vintage into one song_id -> artist table."""
    readers = {"2013": _read_2013, "2014": _read_2014, "2015": _read_2015}
    frames, provenance = [], {}
    for name, reader in readers.items():
        part = reader(META[name])
        assert len(part) > 0, f"{name} metadata parsed to 0 rows ({META[name]})"
        part["source"] = name
        frames.append(part)
        provenance[name] = len(part)

    meta = pd.concat(frames, ignore_index=True)
    meta["artist_raw"] = meta["artist_raw"].replace({"nan": np.nan, "n/a": np.nan, "": np.nan})
    meta = meta.drop_duplicates(subset=["song_id"], keep="first")

    conflicts = (
        meta[meta["artist_raw"].notna()]
        .groupby("song_id")["artist_raw"]
        .nunique()
        .gt(1)
        .sum()
    )
    print(f"[meta] conflicting artist spellings across vintages: {conflicts}")
    return meta, provenance


def attach_artists(targets: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    df = targets.merge(meta[["song_id", "artist_raw", "source"]], on="song_id", how="left")

    for name in META:
        n = int((df["source"] == name).sum())
        print(f"[meta] clips joined to {name} metadata: {n}")

    df["artist"] = df["artist_raw"].map(normalize_artist)
    missing = df["artist"].isna() | df["artist"].eq("")
    # An unidentifiable artist cannot be proven distinct, so give each such
    # clip its own group rather than clumping unrelated songs into one fold.
    df.loc[missing, "artist"] = "unknown-song-" + df.loc[missing, "song_id"].astype(str)
    df["artist_is_unknown"] = missing

    print(f"[meta] clips with no artist metadata: {int(missing.sum())}")
    return df


# --------------------------------------------------------------------------- #
# 3. Splitting
# --------------------------------------------------------------------------- #
def make_folds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("song_id").reset_index(drop=True).copy()
    groups = df["artist"].to_numpy()

    n_groups = len(pd.unique(groups))
    assert n_groups >= N_SPLITS, f"need >= {N_SPLITS} artist groups, have {n_groups}"

    gkf = GroupKFold(n_splits=N_SPLITS)
    fold = np.full(len(df), -1, dtype=int)
    dummy = np.zeros(len(df))
    for k, (_, test_idx) in enumerate(gkf.split(dummy, groups=groups)):
        fold[test_idx] = k
    assert (fold >= 0).all(), "some clips were not assigned to a fold"
    df["fold"] = fold
    return df


def assert_no_artist_overlap(df: pd.DataFrame) -> None:
    """Hard check: no normalized artist may appear in more than one fold."""
    fold_sets = df.groupby("fold")["artist"].agg(set)
    folds = list(fold_sets.index)
    overlaps = []
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            shared = fold_sets[folds[i]] & fold_sets[folds[j]]
            if shared:
                overlaps.append((folds[i], folds[j], sorted(shared)[:5]))
    assert not overlaps, f"artist leakage across folds: {overlaps}"

    max_folds = df.groupby("artist")["fold"].nunique().max()
    assert max_folds == 1, f"an artist spans {max_folds} folds"
    print(f"[check] artist overlap across all {N_SPLITS} folds: 0 (PASS)")


# --------------------------------------------------------------------------- #
# 4. Audit
# --------------------------------------------------------------------------- #
def audit(df: pd.DataFrame, provenance: dict[str, int]) -> None:
    print("\n" + "=" * 62)
    print("DATA AUDIT")
    print("=" * 62)
    print(f"total clips              : {len(df)}")
    print(f"metadata rows by vintage : " + ", ".join(f"{k}={v}" for k, v in provenance.items()))
    print(f"unique artists (raw)     : {df.loc[~df.artist_is_unknown, 'artist_raw'].nunique()}")
    print(f"unique artist groups     : {df['artist'].nunique()} (normalized)")
    print(f"clips w/o artist metadata: {int(df['artist_is_unknown'].sum())}")

    print("\nclips per fold:")
    counts = df["fold"].value_counts().sort_index()
    for k, n in counts.items():
        raw = df.loc[(df.fold == k) & ~df.artist_is_unknown, "artist_raw"].nunique()
        print(f"  fold {k:2d}: {n:4d} clips   ({raw} unique artists)")

    print("\nmissing values in processed targets:")
    cols = ["valence_mean", "arousal_mean", "valence_std", "arousal_std"]
    any_missing = False
    for c in cols:
        n = int(df[c].isna().sum())
        any_missing |= n > 0
        print(f"  {c:<14}: {n}")
    if not any_missing:
        print("  -> none")

    print("\ntarget ranges after rescale:")
    print(df[cols].describe().loc[["min", "max", "mean"]].round(4).to_string())
    print("=" * 62)


# --------------------------------------------------------------------------- #
def main() -> None:
    targets = load_targets()
    targets.to_csv(DATA / "processed_targets.csv", index=False)
    print(f"[write] {DATA / 'processed_targets.csv'} ({len(targets)} rows)")

    meta, provenance = load_artist_map()
    df = attach_artists(targets, meta)
    df = make_folds(df)
    assert_no_artist_overlap(df)

    splits = df[["song_id", "artist", "fold"]].copy()
    splits.to_csv(ROOT / "splits.csv", index=False)
    print(f"[write] {ROOT / 'splits.csv'} ({len(splits)} rows)")

    audit(df, provenance)


if __name__ == "__main__":
    main()
