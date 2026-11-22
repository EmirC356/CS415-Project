"""Data preprocessing, feature aggregation, and dataset loading pipeline.

Handles:
1. Static annotation loading, column cleaning, and linear rescaling to [-1, 1].
2. OpenSMILE frame-level acoustic feature aggregation across time (mean & std).
3. Modular loader for acoustic (openSMILE) and self-supervised (MERT) representations.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_static_annotations_file(data_dir: str = "data") -> str:
    """Find the static annotations CSV file in data_dir."""
    candidates = [
        os.path.join(
            data_dir,
            "DEAM_Annotations",
            "annotations",
            "annotations averaged per song",
            "song_level",
            "static_annotations_averaged_songs_1_2000.csv",
        ),
        os.path.join(data_dir, "static_annotations_averaged_songs_1_2000.csv"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    # Fallback glob
    matches = glob.glob(os.path.join(data_dir, "**", "*static_annotations_averaged_songs_1_2000.csv"), recursive=True)
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not locate static_annotations_averaged_songs_1_2000.csv inside '{data_dir}'"
    )


def find_opensmile_feature_dir(data_dir: str = "data") -> str:
    """Find the directory containing per-song openSMILE CSV feature files."""
    candidates = [
        os.path.join(data_dir, "features", "features"),
        os.path.join(data_dir, "features"),
    ]
    for c in candidates:
        if os.path.isdir(c) and len(glob.glob(os.path.join(c, "*.csv"))) > 100:
            return c

    matches = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    # Group by directory and find one with hundreds of numeric CSVs
    dirs = {}
    for m in matches:
        base = os.path.basename(m)
        name, _ = os.path.splitext(base)
        if name.isdigit():
            d = os.path.dirname(m)
            dirs[d] = dirs.get(d, 0) + 1

    if dirs:
        best_dir = max(dirs.items(), key=lambda kv: kv[1])[0]
        return best_dir

    raise FileNotFoundError(f"Could not locate directory containing openSMILE CSV files in '{data_dir}'")


def preprocess_static_targets(
    annotations_path: Optional[str] = None,
    output_path: str = "data/processed_targets.csv",
    data_dir: str = "data",
) -> pd.DataFrame:
    """Load static annotations for the 1,744 45-second clips and linearly rescale.

    Rescaling:
        valence_mean -> (y - 5.0) / 4.0  (maps [1, 9] to [-1, 1])
        arousal_mean -> (y - 5.0) / 4.0  (maps [1, 9] to [-1, 1])
        valence_std  -> y / 4.0
        arousal_std  -> y / 4.0
    """
    if annotations_path is None:
        annotations_path = find_static_annotations_file(data_dir)

    logger.info("Loading static annotations from: %s", annotations_path)
    df = pd.read_csv(annotations_path)

    # Clean whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    expected_cols = ["song_id", "valence_mean", "arousal_mean", "valence_std", "arousal_std"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns {missing} in {annotations_path}. Found: {df.columns.tolist()}")

    df = df[expected_cols].copy()
    df["song_id"] = df["song_id"].astype(int)

    # Apply specified linear transformations
    df["valence_mean"] = (df["valence_mean"] - 5.0) / 4.0
    df["arousal_mean"] = (df["arousal_mean"] - 5.0) / 4.0
    df["valence_std"] = df["valence_std"] / 4.0
    df["arousal_std"] = df["arousal_std"] / 4.0

    df = df.sort_values("song_id").reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Processed targets saved to %s (Shape: %s)", output_path, df.shape)

    return df


def _aggregate_single_opensmile_file(
    file_path: str,
) -> Optional[Tuple[int, np.ndarray, np.ndarray, List[str]]]:
    """Process a single openSMILE CSV file with polars for high throughput."""
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    try:
        song_id = int(base_name)
    except ValueError:
        return None

    try:
        df = pl.read_csv(file_path, separator=";")
    except Exception as e:
        logger.error("Failed to read %s: %s", file_path, e)
        return None

    feature_cols = [c for c in df.columns if c != "frameTime"]

    # Compute mean and standard deviation per column
    means = df.select([pl.col(c).mean() for c in feature_cols]).to_numpy()[0]
    stds = df.select([pl.col(c).std() for c in feature_cols]).to_numpy()[0]

    # Handle zero standard deviation or rare cases
    stds = np.nan_to_num(stds, nan=0.0)
    means = np.nan_to_num(means, nan=0.0)

    return song_id, means, stds, feature_cols


def aggregate_opensmile_features(
    feature_dir: Optional[str] = None,
    output_path: str = "data/opensmile_aggregated.parquet",
    target_song_ids: Optional[List[int]] = None,
    data_dir: str = "data",
    max_workers: int = 8,
) -> pd.DataFrame:
    """Aggregate per-song frame-level openSMILE CSVs across time using mean and std.

    Results in 260 * 2 = 520 features per song.
    """
    if feature_dir is None:
        feature_dir = find_opensmile_feature_dir(data_dir)

    logger.info("Aggregating openSMILE features from: %s", feature_dir)
    all_files = glob.glob(os.path.join(feature_dir, "*.csv"))

    if target_song_ids is not None:
        target_set = set(target_song_ids)
        file_map = {}
        for f in all_files:
            b = os.path.splitext(os.path.basename(f))[0]
            if b.isdigit() and int(b) in target_set:
                file_map[int(b)] = f
        files_to_process = [file_map[sid] for sid in target_song_ids if sid in file_map]
        logger.info(
            "Found %d of %d requested song files in %s",
            len(files_to_process),
            len(target_song_ids),
            feature_dir,
        )
    else:
        files_to_process = all_files
        logger.info("Processing all %d CSV files found in %s", len(files_to_process), feature_dir)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(_aggregate_single_opensmile_file, files_to_process):
            if res is not None:
                results.append(res)

    results.sort(key=lambda x: x[0])  # Sort by song_id

    song_ids = [r[0] for r in results]
    feature_cols = results[0][3]

    mean_col_names = [f"{col}_mean" for col in feature_cols]
    std_col_names = [f"{col}_std" for col in feature_cols]
    all_col_names = mean_col_names + std_col_names

    matrix_rows = []
    for r in results:
        # Concatenate mean and std
        vector = np.concatenate([r[1], r[2]])
        matrix_rows.append(vector)

    feat_matrix = np.vstack(matrix_rows)

    df_out = pd.DataFrame(feat_matrix, columns=all_col_names)
    df_out.insert(0, "song_id", song_ids)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_out.to_parquet(output_path, index=False)
    logger.info(
        "openSMILE features aggregated and cached to %s (Shape: %s, Features: %d)",
        output_path,
        df_out.shape,
        len(all_col_names),
    )

    return df_out


def load_mert_features(
    song_ids: List[int],
    mert_dir: str = "cache/mert95m",
    layer: int = 7,
    pool: str = "mean",
) -> Tuple[np.ndarray, List[str]]:
    """Load cached MERT embeddings as a clip x features matrix.

    Matches `features/extract_mert.py`, which writes one `{song_id}.npy` per clip
    with shape (n_windows, 13, 768): 13 hidden states, frames mean-pooled inside
    each window. This loader takes one layer, pools the windows across time
    (the contract keeps pooling in model code, not in the cache) and returns
    (N, 768).

    Args:
        layer: index into the 13 hidden states (0 = CNN output, 12 = last layer).
        pool: "mean" or "std" over windows.
    """
    if not 0 <= layer <= 12:
        raise ValueError(f"layer must be 0-12, got {layer}")
    if pool not in ("mean", "std"):
        raise ValueError(f"pool must be 'mean' or 'std', got {pool!r}")

    logger.info("Loading MERT features for %d songs from %s (layer %d, %s-pooled)", len(song_ids), mert_dir, layer, pool)

    vectors = []
    for sid in song_ids:
        path = os.path.join(mert_dir, f"{sid}.npy")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"MERT embedding for song_id {sid} not found at '{path}'. "
                f"Run features/extract_mert.py --out {mert_dir} first."
            )
        emb = np.load(path)  # (n_windows, 13, 768)
        if emb.ndim != 3 or emb.shape[1] != 13 or emb.shape[2] != 768:
            raise ValueError(f"{path}: expected (n_windows, 13, 768), got {emb.shape}")
        per_window = emb[:, layer, :].astype(np.float32)
        vectors.append(per_window.mean(axis=0) if pool == "mean" else per_window.std(axis=0))

    X = np.vstack(vectors).astype(np.float32)
    feature_names = [f"mert_l{layer}_{pool}_{i}" for i in range(X.shape[1])]
    return X, feature_names


LABELS_PATH = "data/interim/labels_fixed.csv"
OPENSMILE_CACHE = "cache/opensmile/clip_features.parquet"


def load_targets(path: str = LABELS_PATH) -> pd.DataFrame:
    """Corrected static targets (2013 V/A swap fixed), rescaled to [-1, 1].

    Built by data/audit/audit_deam.py + data/audit/check_va_swap.py --fix.
    Columns: song_id, valence_mean, arousal_mean, valence_std, arousal_std
    (listener sd s/4), valence_std_adj, arousal_std_adj (rater bias/scale
    adjusted sd), n_raters, batch.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Build it with data/audit/audit_deam.py and "
            "data/audit/check_va_swap.py --fix (see docs/baseline_notes_efe.md)."
        )
    df = pd.read_csv(path)
    if "va_swap_fixed" not in df.columns:
        raise ValueError(f"{path} lacks the va_swap_fixed flag: it is not the corrected label file.")
    out = pd.DataFrame({
        "song_id": df["song_id"].astype(int),
        "valence_mean": df["mu_V"],
        "arousal_mean": df["mu_A"],
        "valence_std": df["s_V"],
        "arousal_std": df["s_A"],
        "valence_std_adj": df["s_V_adj"],
        "arousal_std_adj": df["s_A_adj"],
        "n_raters": df["n_raters"],
        "batch": df["batch"],
    })
    return out.sort_values("song_id").reset_index(drop=True)


def load_predictions(path: str) -> pd.DataFrame:
    """Read a prediction file in either the contract format (results/predictions/,
    columns fold_id, mu_V, mu_A, sigma_V, sigma_A) or the prototype format
    (outputs/, columns fold, mu_v, mu_a, sigma_v, sigma_a); return the prototype names."""
    df = pd.read_csv(path)
    return df.rename(columns={"fold_id": "fold", "mu_V": "mu_v", "mu_A": "mu_a",
                              "sigma_V": "sigma_v", "sigma_A": "sigma_a"})


def load_dataset(
    feature_type: str = "opensmile",
    data_dir: str = "data",
    splits_path: Optional[str] = None,
    mert_dir: str = "cache/mert95m",
    mert_layer: int = 7,
    mert_pool: str = "mean",
    targets_path: str = LABELS_PATH,
    opensmile_path: str = OPENSMILE_CACHE,
) -> Dict[str, Union[np.ndarray, List[str], pd.DataFrame]]:
    """Unified loader providing aligned feature matrix X, targets y_mean, y_std, and song_ids.

    Returns:
        dict with keys:
            - 'song_ids': ndarray (N,)
            - 'X': ndarray (N, D)
            - 'y_mean': ndarray (N, 2)  [valence_mean, arousal_mean]
            - 'y_std': ndarray (N, 2)   [valence_std, arousal_std]
            - 'feature_names': list of str
            - 'splits': pd.DataFrame with song_id and fold (if splits_path provided or found)
            - 'df_targets': pd.DataFrame
    """
    df_targets = load_targets(targets_path)
    target_song_ids = df_targets["song_id"].tolist()

    if feature_type.lower() == "opensmile":
        if not os.path.isfile(opensmile_path):
            raise FileNotFoundError(
                f"{opensmile_path} not found. Build it with features/aggregate_opensmile.py "
                "--labels data/interim/labels_fixed.csv --metadata-2013 data/metadata/metadata_2013.csv"
            )
        df_feat = pd.read_parquet(opensmile_path).loc[target_song_ids]
        X = df_feat.to_numpy(dtype=np.float32)
        feature_names = df_feat.columns.tolist()

    elif feature_type.lower() == "mert":
        X, feature_names = load_mert_features(
            target_song_ids, mert_dir=mert_dir, layer=mert_layer, pool=mert_pool
        )

    else:
        raise ValueError(f"Unknown feature_type: '{feature_type}'. Choose 'opensmile' or 'mert'.")

    y_mean = df_targets[["valence_mean", "arousal_mean"]].to_numpy(dtype=np.float32)
    y_std = df_targets[["valence_std", "arousal_std"]].to_numpy(dtype=np.float32)
    song_ids = df_targets["song_id"].to_numpy(dtype=np.int64)

    # Look for splits file (official contract: splits/folds_artist.csv)
    splits_df = None
    splits_candidates = [
        splits_path,
        "splits/folds_artist.csv",
        os.path.join(data_dir, "folds_artist.csv"),
    ]
    for sp in splits_candidates:
        if sp and os.path.isfile(sp):
            splits_df = pd.read_csv(sp)
            splits_df["song_id"] = splits_df["song_id"].astype(int)
            logger.info("Loaded splits from: %s", sp)
            break

    return {
        "song_ids": song_ids,
        "X": X,
        "y_mean": y_mean,
        "y_std": y_std,
        "feature_names": feature_names,
        "df_targets": df_targets,
        "splits": splits_df,
    }


def main():
    parser = argparse.ArgumentParser(description="Sanity-check the corrected inputs used by src/train.py.")
    parser.add_argument("--labels", default=LABELS_PATH)
    parser.add_argument("--features", default=OPENSMILE_CACHE)
    args = parser.parse_args()
    data = load_dataset(targets_path=args.labels, opensmile_path=args.features)
    logger.info("X %s, y_mean %s, folds loaded: %s", data["X"].shape, data["y_mean"].shape, data["splits"] is not None)


if __name__ == "__main__":
    main()
